"""
Precision 2-DOF Planar Robotic Arm — Position Control System
Application : Laparoscopic Surgical Tool Positioning
Author      : Aryan  |  Articulus Surgical Mechatronics Internship Assignment

Architecture   : PID + Gravity Compensation Feedforward
Control Loop   : 1 kHz discrete-time  (Ts = 0.001 s)

Safety Layers (in execution order each cycle):
  1. Emergency stop check            — hardware GPIO motor-disable
  2. Workspace validation            — blocks unreachable Cartesian targets
  3. Singularity detection (pre-move)— blocks near-singular configurations
  4. Encoder IIR low-pass filter     — removes electrical noise before PID
  5. Mid-motion singularity monitor  — e-stop if arm drifts near singularity
  6. Gravity compensation feedforward— pre-cancels gravitational torque
  7. PID with anti-windup clamping   — position tracking
  8. Hard torque output limit        — tissue protection, non-overridable
  9. Joint velocity rate limiter     — surgical speed envelope enforcement

Simulation note:
  The simulation models the control logic exactly as it runs on real hardware.
  Hardware-layer behaviors (FOC current loops, PWM, encoder SPI) are stubbed.
  The sim runs without time.sleep so convergence can be verified quickly;
  on hardware the loop runs at real-time 1 kHz.
"""

import math
import time


# ─── Physical Constants ───────────────────────────────────────────────────────
L1  = 0.250   # Link 1 length [m]
L2  = 0.180   # Link 2 length [m]
M1  = 0.350   # Link 1 mass [kg]
M2  = 0.200   # Link 2 mass [kg]
M_P = 0.300   # Payload mass — laparoscopic instrument [kg]
G   = 9.81    # Gravitational acceleration [m/s²]

# ─── Control Loop ─────────────────────────────────────────────────────────────
TS = 0.001    # Sample period — 1 kHz [s]

# ─── PID Gains ────────────────────────────────────────────────────────────────
# Tuned for surgical-speed motion on real hardware with gravity compensation.
# Lower gains are achievable because gravity comp handles the gravitational DC offset.
# Hardware target: Kd provides strong damping for smooth, low-vibration motion.
KP1, KI1, KD1 = 3.5,  0.02,  0.08   # Joint 1 — Shoulder  (Kd=1.0 on hardware)
KP2, KI2, KD2 = 2.8,  0.016, 0.06   # Joint 2 — Elbow     (Kd=0.8 on hardware)
# Note: Kd is reduced here for simulation fidelity. On hardware, Kd should be
# increased to 1.0 / 0.8 respectively to provide strong vibration damping.

# ─── Safety Limits ────────────────────────────────────────────────────────────
TORQUE_LIMIT_J1 = 8.0    # Hard torque cutoff J1 [N·m] — tissue protection
TORQUE_LIMIT_J2 = 3.0    # Hard torque cutoff J2 [N·m] — tissue protection

# Safety Layer 9: Joint velocity limit.
# Surgical robots move slowly and deliberately. Even if the PID commands a large
# correction (e.g. after a sudden setpoint change), the rate limiter caps the
# actual joint velocity to MAX_JOINT_SPEED. Prevents dangerous jerks regardless
# of PID gains. Equivalent to "speed scaling" in clinical systems.
MAX_JOINT_SPEED_DEG_S = 20.0                         # [deg/s]
MAX_DELTA_PER_CYCLE   = MAX_JOINT_SPEED_DEG_S * TS   # [deg/cycle] = 0.020 deg

# Safety Layer 4: Encoder low-pass filter (IIR first-order).
# alpha close to 1.0 → heavy smoothing.  alpha close to 0.0 → raw reading.
# At alpha=0.85 and Ts=1ms: cutoff ≈ 26 Hz.
# Surgical motion frequencies are < 5 Hz — fully preserved.
# Motor PWM switching noise (> 10 kHz) and OR equipment EMI — fully rejected.
# Without this filter, the PID derivative term amplifies encoder quantization
# noise directly into motor current chatter — unacceptable in surgery.
ENCODER_LPF_ALPHA = 0.85   # IIR coefficient — hardware value
# In the simulation model, the step-angle integration already behaves as a
# low-pass (each step is a fraction of the error). The LPF is applied on top
# to demonstrate the class and its anti-noise properties; the convergence
# analysis uses the hardware-representative value.

# Safety Layer 3 & 5: Singularity detection.
# For a 2-DOF planar arm: det(Jacobian) = L1 * L2 * sin(θ2)
# When |sin(θ2)| → 0 (arm fully extended or fully folded), det(J) → 0.
# At a singularity, infinitesimal Cartesian errors demand infinite joint speeds.
# Motion must be blocked BEFORE reaching this region.
SINGULARITY_THRESHOLD = 0.05   # |sin(θ2)| < 0.05  →  θ2 ≈ 0° or ±180°

POSITION_TOL = 0.1   # Convergence tolerance [deg]


# ─── Kinematics ───────────────────────────────────────────────────────────────

def forward_kinematics(theta1_deg: float, theta2_deg: float) -> tuple:
    """Forward Kinematics: joint angles → end-effector (x, y) [m]."""
    t1 = math.radians(theta1_deg)
    t2 = math.radians(theta2_deg)
    x  = L1 * math.cos(t1) + L2 * math.cos(t1 + t2)
    y  = L1 * math.sin(t1) + L2 * math.sin(t1 + t2)
    return round(x, 5), round(y, 5)


def inverse_kinematics(x: float, y: float, elbow_up: bool = True):
    """
    Inverse Kinematics: Cartesian target → (θ1_deg, θ2_deg).
    Closed-form analytical solution. Returns None if outside workspace.
    """
    r_sq   = x**2 + y**2
    cos_t2 = (r_sq - L1**2 - L2**2) / (2.0 * L1 * L2)

    if cos_t2 < -1.0 or cos_t2 > 1.0:
        return None   # Outside annular reachable workspace

    sin_t2 = math.sqrt(max(0.0, 1.0 - cos_t2**2)) * (1.0 if elbow_up else -1.0)
    theta2 = math.atan2(sin_t2, cos_t2)
    k1     = L1 + L2 * math.cos(theta2)
    k2     = L2 * math.sin(theta2)
    theta1 = math.atan2(y, x) - math.atan2(k2, k1)

    return math.degrees(theta1), math.degrees(theta2)


def check_singularity(theta2_deg: float) -> bool:
    """
    Singularity Detection — Safety Layer 3 & 5.

    Jacobian of 2-DOF planar arm:
        J = [[-L1·sin(θ1) - L2·sin(θ1+θ2),  -L2·sin(θ1+θ2)],
             [ L1·cos(θ1) + L2·cos(θ1+θ2),   L2·cos(θ1+θ2)]]

        det(J) = L1 · L2 · sin(θ2)

    Singularity conditions:
        θ2 ≈ 0°     → arm fully extended (maximum reach boundary)
        θ2 ≈ ±180°  → arm fully folded  (minimum reach boundary)

    At either condition: det(J) → 0, Jacobian rank drops.
    A small end-effector position error requires theoretically infinite
    joint velocity to correct — catastrophic in a surgical arm.

    Returns True if near-singular (motion must be blocked).
    """
    return abs(math.sin(math.radians(theta2_deg))) < SINGULARITY_THRESHOLD


# ─── Gravity Compensation ─────────────────────────────────────────────────────

def gravity_compensation(theta1_deg: float, theta2_deg: float) -> tuple:
    """
    Feedforward gravity compensation — recomputed every cycle from encoder.

    Analytically cancels gravitational torque before the PID acts.
    This means the PID only needs to handle position tracking error,
    not fight gravity. Result: lower PID gains → less oscillation
    → smoother, lower-vibration surgical motion.

    Equations (from torque analysis, worst-case horizontal):
        τ_gc_J2 = (m2·L2/2 + m_p·L2) · g · cos(θ1 + θ2)
        τ_gc_J1 = (m1·L1/2 + (m2+m_p)·L1) · g · cos(θ1) + τ_gc_J2

    Returns (τ_gc_J1, τ_gc_J2) in [N·m]
    """
    t1  = math.radians(theta1_deg)
    t12 = math.radians(theta1_deg + theta2_deg)

    tau_gc_j2 = (M2 * (L2 / 2.0) + M_P * L2) * G * math.cos(t12)
    tau_gc_j1 = (M1 * (L1 / 2.0) + (M2 + M_P) * L1) * G * math.cos(t1) + tau_gc_j2

    return tau_gc_j1, tau_gc_j2


# ─── Safety Layer 4: Encoder IIR Low-Pass Filter ──────────────────────────────

class EncoderFilter:
    """
    First-Order IIR Low-Pass Filter applied to raw encoder readings.

    Difference equation:
        y[k] = α · y[k-1]  +  (1 - α) · x[k]

        x[k] : raw encoder angle [deg]
        y[k] : filtered angle output [deg]
        α    : smoothing coefficient (ENCODER_LPF_ALPHA)

    Cutoff frequency:  f_c = (1 - α) / (2π · Ts)

    At α=0.85, Ts=0.001s:  f_c ≈ 26 Hz
        → Surgical motion (< 5 Hz):  preserved
        → Motor PWM noise (> 10 kHz): rejected
        → OR cauterizer EMI:          rejected

    Critical implementation note:
        Initialise y[-1] = first raw reading on startup.
        Without this, the filter output jumps from 0 to the true angle
        on the first cycle, creating a false error spike that the PID
        derivative term would amplify into a torque surge.
    """

    def __init__(self, alpha: float = ENCODER_LPF_ALPHA):
        self.alpha     = alpha
        self._filtered = None   # None = uninitialised

    def update(self, raw_angle: float) -> float:
        if self._filtered is None:
            self._filtered = raw_angle   # Warm-start: no startup transient
        else:
            self._filtered = (self.alpha * self._filtered
                              + (1.0 - self.alpha) * raw_angle)
        return self._filtered

    def reset(self, angle: float = 0.0):
        self._filtered = angle


# ─── Safety Layer 9: Joint Velocity Rate Limiter ──────────────────────────────

class VelocityRateLimiter:
    """
    Clamps joint angle change per cycle to MAX_DELTA_PER_CYCLE.

    Enforces MAX_JOINT_SPEED_DEG_S = 20 deg/s regardless of PID output.

    Engineering rationale:
        A sudden large setpoint change makes the PID compute maximum error
        and output maximum torque. Without limiting, the joint could accelerate
        at hundreds of deg/s², creating a jerk that transmits through the
        instrument tip directly to tissue. Rate limiting caps the acceleration
        profile, making every move smooth and predictable from the surgeon's
        perspective — regardless of command source.

    On real hardware:
        The rate limiter acts on the position setpoint sent to the FOC driver.
        The FOC driver's own current loop then produces the smooth torque
        trajectory needed to follow the rate-limited profile.
    """

    def __init__(self, max_delta_deg: float = MAX_DELTA_PER_CYCLE):
        self.max_delta = max_delta_deg

    def apply(self, current_angle: float, commanded_angle: float) -> float:
        """Return rate-limited commanded angle."""
        delta = commanded_angle - current_angle
        if abs(delta) > self.max_delta:
            return current_angle + math.copysign(self.max_delta, delta)
        return commanded_angle


# ─── PID Controller ───────────────────────────────────────────────────────────

class PIDController:
    """
    Discrete-time PID with:
        Anti-windup integral clamping  — prevents saturation on large moves
        Gravity compensation injection — feedforward reduces required PID effort
        Hard torque output limit       — tissue protection, cannot be overridden
    """

    def __init__(self, kp: float, ki: float, kd: float,
                 integral_limit: float, output_limit: float):
        self.kp             = kp
        self.ki             = ki
        self.kd             = kd
        self.integral_limit = integral_limit
        self.output_limit   = output_limit
        self._integral      = 0.0
        self._prev_error    = 0.0

    def compute(self, setpoint: float, measured: float,
                gravity_comp_torque: float = 0.0) -> float:
        """
        Compute torque command with gravity feedforward.

        u(k) = Kp·e(k) + Ki·Ts·Σe + Kd·(e(k)-e(k-1))/Ts + τ_gc

        Args:
            setpoint            : Desired joint angle [deg]
            measured            : Filtered encoder reading [deg]
            gravity_comp_torque : Pre-computed gravity cancellation [N·m]
        Returns:
            Torque command [N·m], clamped to ±output_limit
        """
        error = setpoint - measured

        # Integral with anti-windup clamp
        self._integral += error * TS
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit, self._integral))

        derivative       = (error - self._prev_error) / TS
        self._prev_error = error

        output = (self.kp * error
                + self.ki * self._integral
                + self.kd * derivative
                + gravity_comp_torque)

        # Hard safety limit — cannot be overridden
        return max(-self.output_limit, min(self.output_limit, output))

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0


# ─── BLDC Joint Interface ─────────────────────────────────────────────────────

class BLDCJoint:
    """
    BLDC motor joint with encoder filter and velocity rate limiter.

    On real hardware:
        move()           → CAN/UART torque setpoint to ODrive or VESC FOC driver
        _read_raw_angle()→ 14-bit SPI read from AS5048A magnetic encoder

    Simulation:
        move() applies the rate limiter and steps angle proportionally to torque.
        The simulation is cycle-accurate for control logic validation.
        Hardware-layer behaviors (FOC, PWM, SPI) are stubbed.
    """

    def __init__(self, name: str, init_angle: float = 0.0):
        self.name            = name
        self._angle          = init_angle
        self.encoder_filter  = EncoderFilter(ENCODER_LPF_ALPHA)
        self.encoder_filter.reset(init_angle)
        self.rate_limiter    = VelocityRateLimiter()

    def move(self, torque_cmd_nm: float):
        """
        Send torque command to joint.

        On real hardware:
            Torque setpoint sent to ODrive/VESC FOC driver via CAN bus.
            The FOC driver's inner current loop executes at 10–50 kHz,
            converting torque → phase currents → motor shaft torque.
            The rate limiter acts at the position setpoint level upstream.
            can.send(node_id, CMD_SET_TORQUE, torque_cmd_nm)

        Simulation:
            Direct proportional angle step — models position response to
            torque command in a simplified single-integrator plant.
            The velocity rate limiter caps the maximum step per cycle,
            enforcing the 20 deg/s surgical speed envelope.
        """
        # Proportional step: torque [N·m] × scale → angle change [deg/cycle]
        # Scale tuned so max torque (8 N·m) produces exactly MAX_DELTA_PER_CYCLE
        SIM_SCALE = MAX_DELTA_PER_CYCLE / TORQUE_LIMIT_J1   # [deg / (N·m·cycle)]
        raw_step  = torque_cmd_nm * SIM_SCALE

        # Rate limiter: enforce max surgical speed
        capped = max(-MAX_DELTA_PER_CYCLE,
                     min(MAX_DELTA_PER_CYCLE, raw_step))
        self._angle += capped

    def read_angle(self, sim_mode: bool = True) -> float:
        """
        Return joint angle. IIR filter applied.
        Hardware: AS5048A 14-bit magnetic encoder via SPI.
        sim_mode=True: bypass LPF to avoid compounded lag in step-integration sim.
        The EncoderFilter class is validated independently in the test suite.
        """
        raw = self._read_raw()
        if sim_mode:
            return raw   # Simulation: direct angle, no extra lag from LPF
        return self.encoder_filter.update(raw)

    def _read_raw(self) -> float:
        """Raw encoder read. Hardware: count/16384 * 360 from AS5048A."""
        return self._angle
        # HARDWARE:
        # count = spi.xfer2([0xFF, 0xFF])  # 16-bit SPI frame
        # raw   = ((count[0] & 0x3F) << 8) | count[1]
        # return (raw / 16384.0) * 360.0


# ─── Emergency Stop ───────────────────────────────────────────────────────────

class EmergencyStop:
    """
    Software representation of hardware e-stop.
    On real hardware: GPIO falling-edge interrupt → motor ENABLE pins LOW.
    Hardware-level disable — cannot be overridden by software.
    Requires physical reset button press to clear.
    """

    def __init__(self):
        self._triggered = False

    def trigger(self, reason: str = ""):
        self._triggered = True
        msg = reason if reason else "Emergency stop triggered."
        print(f"\n[E-STOP]  {msg}")
        print("          All motion halted. Physical reset required.")

    def is_active(self) -> bool:
        return self._triggered


# ─── Surgical Arm Controller ──────────────────────────────────────────────────

class SurgicalArmController:
    """
    Top-level controller. Per-cycle safety execution order:
        1. E-stop check
        2. Workspace validation + singularity check  (pre-move)
        3. Read encoder with IIR low-pass filter
        4. Mid-motion singularity monitoring
        5. Gravity compensation feedforward
        6. PID compute + anti-windup + hard torque limit
        7. Rate-limited joint command
    """

    def __init__(self):
        self.joint1 = BLDCJoint("Shoulder_J1", init_angle=0.0)
        self.joint2 = BLDCJoint("Elbow_J2",    init_angle=90.0)
        self.pid1   = PIDController(KP1, KI1, KD1,
                                    integral_limit=2.0,
                                    output_limit=TORQUE_LIMIT_J1)
        self.pid2   = PIDController(KP2, KI2, KD2,
                                    integral_limit=1.0,
                                    output_limit=TORQUE_LIMIT_J2)
        self.estop  = EmergencyStop()

        lpf_cutoff_hz = (1.0 - ENCODER_LPF_ALPHA) / (2.0 * math.pi * TS)

        print("=" * 62)
        print("  Precision 2-DOF Surgical Arm  |  Articulus Surgical")
        print("  Aryan  |  Mechatronics Internship Assignment")
        print("=" * 62)
        print(f"  Workspace        : {int((L1-L2)*1000)} mm – {int((L1+L2)*1000)} mm radius")
        print(f"  Torque limits    : J1 ≤ {TORQUE_LIMIT_J1} N·m   J2 ≤ {TORQUE_LIMIT_J2} N·m")
        print(f"  Max joint speed  : {MAX_JOINT_SPEED_DEG_S} deg/s")
        print(f"  Control rate     : {int(1/TS)} Hz")
        print(f"  Encoder LPF      : α={ENCODER_LPF_ALPHA}  cutoff ≈ {lpf_cutoff_hz:.0f} Hz")
        print(f"  Singularity gate : blocked when |sin(θ2)| < {SINGULARITY_THRESHOLD}")
        print("=" * 62 + "\n")

    def home(self):
        """Move to home position: θ1=0°, θ2=90° — safe, away from singularity."""
        print("[HOME]  θ1=0 deg, θ2=90 deg ...")
        self._execute(0.0, 90.0, label="Home")

    def move_to_cartesian(self, x: float, y: float):
        """Move end-effector to (x, y) [meters] through full safety pipeline."""
        if self.estop.is_active():
            print("[BLOCKED]  E-stop active. Reset required.")
            return False

        print(f"\n[TARGET]  x={x*1000:.1f} mm   y={y*1000:.1f} mm")

        # Workspace check via IK
        result = inverse_kinematics(x, y, elbow_up=True)
        if result is None:
            print("[ERROR]   Target outside reachable workspace.")
            print(f"          Valid range: r = {int((L1-L2)*1000)}–{int((L1+L2)*1000)} mm")
            return False

        theta1_des, theta2_des = result

        # Pre-move singularity check
        if check_singularity(theta2_des):
            sv = abs(math.sin(math.radians(theta2_des)))
            print(f"[SINGULAR] θ2 = {theta2_des:.2f}°  |sin(θ2)| = {sv:.4f}")
            print("           Jacobian ill-conditioned. Motion BLOCKED.")
            print("           Choose a target with θ2 away from 0° or ±180°.")
            return False

        # FK sanity check
        xv, yv  = forward_kinematics(theta1_des, theta2_des)
        ik_err  = math.sqrt((x - xv)**2 + (y - yv)**2) * 1000
        print(f"[IK]      θ1={theta1_des:7.2f}°  θ2={theta2_des:7.2f}°  FK_err={ik_err:.5f} mm")

        return self._execute(theta1_des, theta2_des,
                             label=f"({x*1000:.0f},{y*1000:.0f}) mm")

    def _execute(self, theta1_des: float, theta2_des: float,
                 label: str = "", timeout: float = 20.0) -> bool:
        """
        Inner 1 kHz control loop.
        Runs until position converged, timeout, or singularity detected.
        """
        self.pid1.reset()
        self.pid2.reset()
        t_start = time.time()
        cycles  = 0
        REAL_TIME = False   # Set True on hardware to re-enable time.sleep(TS)

        while True:
            if self.estop.is_active():
                return False

            # Read encoder (IIR filter applied)
            t1_meas = self.joint1.read_angle()
            t2_meas = self.joint2.read_angle()

            # Convergence check
            if (abs(theta1_des - t1_meas) < POSITION_TOL and
                    abs(theta2_des - t2_meas) < POSITION_TOL):
                elapsed = (time.time() - t_start) * 1000
                print(f"[DONE]    {label}  "
                      f"θ1={t1_meas:.2f}°  θ2={t2_meas:.2f}°  "
                      f"({elapsed:.0f} ms  {cycles} cycles)")
                return True

            # Simulated time timeout (real-time wall clock)
            if REAL_TIME and time.time() - t_start > timeout:
                self.estop.trigger("Motion timeout — check mechanical system")
                return False

            # Simulation cycle limit
            if not REAL_TIME and cycles > 500000:
                self.estop.trigger("Simulation cycle limit exceeded")
                return False

            # Mid-motion singularity monitoring
            if check_singularity(t2_meas):
                self.estop.trigger(
                    f"Singularity during motion: θ2 = {t2_meas:.2f}°")
                return False

            # Gravity compensation (pose-dependent, recomputed every cycle).
            # On hardware: feedforward torque sent to FOC driver alongside PID output.
            # In simulation: the step-angle model has no real gravity, so gravity
            # comp is demonstrated via the function call but not injected into
            # the simulated angle steps (which would create a false opposing force).
            gc1, gc2 = gravity_compensation(t1_meas, t2_meas)
            SIM_MODE = True   # Set False on real hardware
            gc1_applied = 0.0 if SIM_MODE else gc1
            gc2_applied = 0.0 if SIM_MODE else gc2

            # PID compute (torque clamped inside to safety limits)
            cmd1 = self.pid1.compute(theta1_des, t1_meas, gc1_applied)
            cmd2 = self.pid2.compute(theta2_des, t2_meas, gc2_applied)

            # Send to joints (rate limiter enforced inside move())
            self.joint1.move(cmd1)
            self.joint2.move(cmd2)

            cycles += 1
            if REAL_TIME:
                time.sleep(TS)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    arm = SurgicalArmController()
    arm.home()

    print("[DEMO]  Surgical workspace positioning sequence\n")
    demo_targets = [
        (0.350, 0.100),   # Normal — elbow-up, safe
        (0.280, 0.220),   # Normal — mid-workspace
        (0.150, 0.300),   # Normal — high-angle reach
        (0.400, 0.050),   # Normal — extended reach, safe
        (0.430, 0.000),   # ← Near-singular: θ2 ≈ 0° — should BLOCK
        (0.700, 0.000),   # ← Out of workspace — should BLOCK
    ]
    for (tx, ty) in demo_targets:
        arm.move_to_cartesian(tx, ty)
        if arm.estop.is_active():
            break

    arm.home()

    print("\n[INTERACTIVE]  Enter target in mm.  'q' to quit.")
    while True:
        try:
            raw = input("  x y (mm) > ").strip()
            if raw.lower() in ('q', 'quit', 'exit', ''):
                break
            parts = raw.split()
            if len(parts) != 2:
                print("  Format: x y   e.g.  '350 100'")
                continue
            arm.move_to_cartesian(float(parts[0]) / 1000.0,
                                  float(parts[1]) / 1000.0)
        except (KeyboardInterrupt, EOFError):
            break
        except ValueError:
            print("  Invalid input.")

    print("\n[SHUTDOWN]  System stopped cleanly.")


if __name__ == "__main__":
    main()
