import streamlit as st
import cmath
import random
import math
import time
import plotly.graph_objects as go # Using Plotly for better web visualization

# --- Quantum Core Logic (Mostly unchanged) ---

# (We can keep the Qubit class largely the same, but operations will
# modify the state stored in Streamlit's session state instead of an object instance)

def format_complex(z):
    """Formats complex numbers nicely."""
    if not isinstance(z, complex): z = complex(z) # Ensure it's complex
    if abs(z.real) < 1e-9 and abs(z.imag) < 1e-9: return "0.000"
    if abs(z.imag) < 1e-9: return f"{z.real:.3f}"
    if abs(z.real) < 1e-9: return f"{z.imag:.3f}j"
    return f"{z.real:.3f}{z.imag:+.3f}j"

def normalize_state(alpha, beta):
    """Normalizes the given alpha and beta amplitudes."""
    norm = abs(alpha)**2 + abs(beta)**2
    if not math.isclose(norm, 1.0, abs_tol=1e-9):
        if norm == 0:
             return 1+0j, 0+0j # Default to |0> if input is zero vector
        else:
            sqrt_norm = math.sqrt(norm)
            return alpha / sqrt_norm, beta / sqrt_norm
    return alpha, beta

def get_probabilities(alpha, beta):
    """Calculates probabilities from amplitudes."""
    prob0 = abs(alpha)**2
    prob1 = abs(beta)**2
    # Ensure sum is close to 1 due to potential floating point inaccuracies
    prob0 = max(0.0, min(1.0, prob0))
    prob1 = 1.0 - prob0
    return prob0, prob1

def apply_gate_to_state(gate_name, alpha, beta):
    """Applies a gate to the given state amplitudes."""
    alpha, beta = complex(alpha), complex(beta) # Ensure complex type
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    new_alpha, new_beta = alpha, beta

    if gate_name == 'H': # Hadamard
        new_alpha = sqrt2_inv * (alpha + beta)
        new_beta = sqrt2_inv * (alpha - beta)
    elif gate_name == 'X': # Pauli-X
        new_alpha = beta
        new_beta = alpha
    elif gate_name == 'Z': # Pauli-Z
        new_alpha = alpha
        new_beta = -beta
    elif gate_name == 'S': # Phase Gate
        new_alpha = alpha
        new_beta = 1j * beta
    elif gate_name == 'T': # pi/8 Gate
        new_alpha = alpha
        new_beta = cmath.exp(1j * math.pi / 4) * beta

    # Return normalized state
    return normalize_state(new_alpha, new_beta)

def apply_decoherence_to_state(alpha, beta, qubit_type, last_change_time, strength=0.05, time_decay_factor=0.1):
    """Applies decoherence to the state amplitudes."""
    alpha, beta = complex(alpha), complex(beta)
    time_elapsed = time.time() - last_change_time
    effective_strength = strength
    if qubit_type == "Majorana-Inspired":
        effective_strength *= 0.1 # More robust

    decay_amount = effective_strength * time_elapsed * time_decay_factor
    decay_factor = max(0, 1.0 - decay_amount)

    prob0, prob1 = get_probabilities(alpha, beta)

    # Don't apply if already in a basis state
    if math.isclose(prob0, 1.0, abs_tol=1e-6) or math.isclose(prob1, 1.0, abs_tol=1e-6):
         return alpha, beta, False # Return original state, indicate no change

    # Simplified model: Reduce the amplitude of the less probable state
    target_alpha, target_beta = alpha, beta
    changed = False
    if prob0 > prob1: # Closer to |0>
        new_target_beta = target_beta * decay_factor
        # Renormalize alpha while preserving phase (handle alpha=0 case)
        if abs(alpha) > 1e-9:
           new_target_alpha = cmath.sqrt(1 - abs(new_target_beta)**2) * (alpha / abs(alpha))
        else: # if alpha was zero, it remains zero unless beta becomes zero
           new_target_alpha = cmath.sqrt(1 - abs(new_target_beta)**2)
        if not cmath.isclose(beta, new_target_beta): changed = True

    else: # Closer to |1> or equal superposition
        new_target_alpha = target_alpha * decay_factor
        # Renormalize beta while preserving phase (handle beta=0 case)
        if abs(beta) > 1e-9:
            new_target_beta = cmath.sqrt(1 - abs(new_target_alpha)**2) * (beta / abs(beta))
        else: # if beta was zero, it remains zero unless alpha becomes zero
            new_target_beta = cmath.sqrt(1 - abs(new_target_alpha)**2)

        if not cmath.isclose(alpha, new_target_alpha): changed = True

    if changed:
        # print(f"Decoherence applied (decay ~{decay_factor:.3f})") # Debugging
        return normalize_state(new_target_alpha, new_target_beta)[0], normalize_state(new_target_alpha, new_target_beta)[1], True
    else:
        return alpha, beta, False


# --- Streamlit App UI and Logic ---

st.set_page_config(layout="wide")
st.title("🌐 Quantum Playground Web App")
st.markdown("Explore single qubit concepts inspired by quantum physics.")

# --- Initialize Session State ---
# Streamlit reruns the script on each interaction, so we use session_state
# to preserve the qubit's state between interactions.

if 'alpha' not in st.session_state:
    st.session_state.alpha = 1+0j
    st.session_state.beta = 0+0j
    st.session_state.qubit_type = "Standard"
    st.session_state.last_measurement = "N/A"
    st.session_state.state_change_time = time.time()
    st.session_state.decoherence_enabled = False # Start disabled
    st.session_state.log = ["App Initialized."] # Keep a log of actions


# --- Helper function to add log messages ---
def add_log(message):
    st.session_state.log.insert(0, f"{time.strftime('%H:%M:%S')}: {message}")
    # Keep log concise
    if len(st.session_state.log) > 10:
        st.session_state.log.pop()

# --- Sidebar Controls ---
st.sidebar.header("Controls")

# Qubit Type
new_qubit_type = st.sidebar.radio(
    "Qubit Type:",
    ("Standard", "Majorana-Inspired"),
    index=("Standard", "Majorana-Inspired").index(st.session_state.qubit_type),
    help="Majorana-Inspired qubits are simulated to be more resistant to decoherence."
)
if new_qubit_type != st.session_state.qubit_type:
    st.session_state.qubit_type = new_qubit_type
    st.session_state.state_change_time = time.time() # Reset timer on type change
    add_log(f"Qubit type changed to {new_qubit_type}.")
    st.experimental_rerun() # Rerun to reflect change immediately

# Decoherence Toggle
st.session_state.decoherence_enabled = st.sidebar.checkbox(
    "Enable Simple Decoherence",
    value=st.session_state.decoherence_enabled,
    help="If enabled, the qubit state will slowly drift towards |0> or |1> over time between interactions."
)


# --- Decoherence Application Point ---
# Apply decoherence based on time elapsed *before* processing button clicks
deco_applied_msg = ""
if st.session_state.decoherence_enabled:
    new_alpha, new_beta, changed = apply_decoherence_to_state(
        st.session_state.alpha,
        st.session_state.beta,
        st.session_state.qubit_type,
        st.session_state.state_change_time
    )
    if changed:
        st.session_state.alpha = new_alpha
        st.session_state.beta = new_beta
        # Don't reset state_change_time here, decoherence is continuous
        deco_applied_msg = "(Decoherence applied) "
        # No add_log here, too verbose. Log specific actions instead.

# --- Initialization Buttons ---
st.sidebar.subheader("Initialize State")
col_init1, col_init2 = st.sidebar.columns(2)
if col_init1.button("Set to |0⟩"):
    st.session_state.alpha, st.session_state.beta = normalize_state(1+0j, 0+0j)
    st.session_state.last_measurement = "N/A"
    st.session_state.state_change_time = time.time()
    add_log("Initialized state to |0⟩.")
    st.experimental_rerun()

if col_init2.button("Set to |1⟩"):
    st.session_state.alpha, st.session_state.beta = normalize_state(0+0j, 1+0j)
    st.session_state.last_measurement = "N/A"
    st.session_state.state_change_time = time.time()
    add_log("Initialized state to |1⟩.")
    st.experimental_rerun()

if col_init1.button("Set to |+⟩"):
    st.session_state.alpha, st.session_state.beta = normalize_state(1/math.sqrt(2), 1/math.sqrt(2))
    st.session_state.last_measurement = "N/A"
    st.session_state.state_change_time = time.time()
    add_log("Initialized state to |+⟩.")
    st.experimental_rerun()

if col_init2.button("Set to |-⟩"):
    st.session_state.alpha, st.session_state.beta = normalize_state(1/math.sqrt(2), -1/math.sqrt(2))
    st.session_state.last_measurement = "N/A"
    st.session_state.state_change_time = time.time()
    add_log("Initialized state to |-⟩.")
    st.experimental_rerun()


# --- Gate Buttons ---
st.sidebar.subheader("Apply Gates")
col_gate1, col_gate2 = st.sidebar.columns(2)

gate_map = {
    "H (Hadamard)": "H", "X (Pauli-X)": "X",
    "Z (Pauli-Z)": "Z", "S (Phase)": "S",
    "T (π/8)": "T"
}
buttons_col1 = ["H (Hadamard)", "Z (Pauli-Z)", "T (π/8)"]
buttons_col2 = ["X (Pauli-X)", "S (Phase)"]

for btn_text in buttons_col1:
    if col_gate1.button(btn_text):
        gate_name = gate_map[btn_text]
        st.session_state.alpha, st.session_state.beta = apply_gate_to_state(
            gate_name, st.session_state.alpha, st.session_state.beta
        )
        st.session_state.last_measurement = "N/A" # Gate applied
        st.session_state.state_change_time = time.time()
        add_log(f"{deco_applied_msg}Applied gate: {gate_name}.")
        st.experimental_rerun()

for btn_text in buttons_col2:
     if col_gate2.button(btn_text):
        gate_name = gate_map[btn_text]
        st.session_state.alpha, st.session_state.beta = apply_gate_to_state(
            gate_name, st.session_state.alpha, st.session_state.beta
        )
        st.session_state.last_measurement = "N/A" # Gate applied
        st.session_state.state_change_time = time.time()
        add_log(f"{deco_applied_msg}Applied gate: {gate_name}.")
        st.experimental_rerun()


# --- Measurement Button ---
st.sidebar.subheader("Measurement")
if st.sidebar.button("Measure Qubit"):
    prob0, prob1 = get_probabilities(st.session_state.alpha, st.session_state.beta)
    outcome = 0 if random.random() < prob0 else 1
    add_log(f"{deco_applied_msg}Measured: P(0)={prob0:.3f}, P(1)={prob1:.3f}. Outcome: {outcome}.")

    # Collapse state
    if outcome == 0:
        st.session_state.alpha, st.session_state.beta = 1+0j, 0+0j
    else:
        st.session_state.alpha, st.session_state.beta = 0+0j, 1+0j

    st.session_state.last_measurement = f"|{outcome}⟩"
    st.session_state.state_change_time = time.time()
    st.experimental_rerun()


# --- Main Display Area ---
col_status, col_viz = st.columns([1, 2]) # Give more space to visualization

with col_status:
    st.subheader("Qubit Status")
    st.metric("Qubit Type", st.session_state.qubit_type)

    st.write("**State Vector:** α|0⟩ + β|1⟩")
    # Use st.code for better monospace formatting
    st.code(f"α = {format_complex(st.session_state.alpha)}\nβ = {format_complex(st.session_state.beta)}", language=None)

    prob0, prob1 = get_probabilities(st.session_state.alpha, st.session_state.beta)
    st.write("**Probabilities:**")
    st.metric("P(Measure |0⟩)", f"{prob0:.4f}")
    st.metric("P(Measure |1⟩)", f"{prob1:.4f}")

    st.metric("Last Measurement", st.session_state.last_measurement)


with col_viz:
    st.subheader("Visualization")

    # Create Plotly Bar Chart for probabilities
    fig = go.Figure(
        data=[
            go.Bar(name='P(0)', x=['|0⟩'], y=[prob0], marker_color='lightblue', text=f"{prob0:.3f}", textposition='outside'),
            go.Bar(name='P(1)', x=['|1⟩'], y=[prob1], marker_color='lightcoral', text=f"{prob1:.3f}", textposition='outside')
        ]
    )
    fig.update_layout(
        yaxis_range=[0, 1.1], # Ensure y-axis goes to 1
        yaxis_title="Probability",
        xaxis_title="Basis State",
        title_text=f"Probability Distribution (Type: {st.session_state.qubit_type})",
        title_x=0.5,
        bargap=0.3, # Gap between bars of different categories (only one category here)
        bargroupgap=0.1 # Gap between bars within the same category
    )
    fig.update_yaxes(ticksuffix=" ") # Add a space to prevent label overlap if value is 1.0

    st.plotly_chart(fig, use_container_width=True)

    # --- Log Display ---
    st.subheader("Action Log")
    st.text_area("", value="\n".join(st.session_state.log), height=150, disabled=True)


# --- Explanations ---
with st.expander("How to Use & Concepts", expanded=False):
    st.markdown("""
    *   **Controls (Sidebar):** Use the buttons to manipulate the qubit.
    *   **Qubit Type:** Choose between a standard qubit and one simulated to be more robust against noise ('Majorana-Inspired').
    *   **Initialize State:** Reset the qubit to common states like |0⟩, |1⟩, or superposition states |+⟩, |-⟩.
    *   **Apply Gates:** Perform quantum operations (H, X, Z, S, T) on the current state.
    *   **Measure Qubit:** Simulate measuring the qubit in the computational basis (|0⟩ or |1⟩). This collapses the superposition according to the probabilities P(0)=|α|² and P(1)=|β|².
    *   **Enable Simple Decoherence:** When checked, the qubit's superposition state will gradually decay towards |0⟩ or |1⟩ between your interactions, simulating environmental noise. Notice the difference in decay speed between the 'Standard' and 'Majorana-Inspired' types.
    *   **Status:** Shows the current complex amplitudes (α, β), the resulting measurement probabilities, and the outcome of the last measurement.
    *   **Visualization:** The bar chart shows the probabilities of measuring |0⟩ or |1⟩.
    *   **Inspiration:** The 'Majorana-Inspired' robustness demonstrates the *goal* of topological quantum computing – creating qubits less prone to errors, although the underlying physics simulation here is highly simplified.
    """)