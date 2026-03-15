# =============================================================================
# Dynamic Pricing Agent — Streamlit Web Application
# =============================================================================
# This app implements a non-cooperative oligopoly pricing engine using
# Game Theory (differentiated Bertrand competition) with a Circuit Breaker
# for predatory pricing scenarios. It delegates strategy generation to the
# OpenAI API and visualizes the results with an interactive bar chart.
# =============================================================================

# --- Standard Library Imports ------------------------------------------------

import json  # Used to parse the structured JSON response from the OpenAI API

# --- Third-Party Imports -----------------------------------------------------

import streamlit as st  # Core framework for building the interactive web UI
from openai import OpenAI  # Official OpenAI Python client (v1+ interface)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# Sets the browser tab title, page icon, and default layout width.
# "wide" layout uses the full browser width for better chart visibility.
# =============================================================================

st.set_page_config(
    page_title="Dynamic Pricing Agent",  # Text shown in the browser tab
    page_icon="📊",  # Emoji icon displayed next to the tab title
    layout="wide",  # Use the full width of the browser viewport
)

# =============================================================================
# SYSTEM PROMPT — PRICING FRAMEWORK DEFINITION
# =============================================================================
# This is the complete set of rules the LLM must follow when generating a
# pricing strategy. It encodes all four constraints from our pricing engine:
#   1. Baseline (P* > MC)
#   2. Game Theory (Nash Equilibrium via Bertrand reaction function)
#   3. Elasticity Adjustment (λ calibration)
#   4. Circuit Breaker (predatory pricing defense)
# The LLM is instructed to return ONLY valid JSON — no markdown, no prose.
# =============================================================================

SYSTEM_PROMPT = """You are a Chief Pricing Strategist specializing in the Digital Economy and Microeconomics.
Your objective is to calculate the optimal dynamic price in a non-cooperative oligopoly market.

PRICING ENGINE — DECISION TREE:

  IF Competitor_Price < MC:
    → CIRCUIT BREAKER triggered.
    → ABORT the Bertrand reaction function entirely.
    → Set Optimal Price = MC × 1.05 (defensive baseline for minimal gross margin).
    → Strategy: "Differentiation Escape" — abandon the loss-making low-end market,
      pivot to product/service differentiation targeting inelastic premium demand.

  ELSE:
    → Apply the differentiated Bertrand reaction function:
      P* = MC + λ × (Competitor_Price − MC)
    → λ (markup factor) is calibrated by Market Price Elasticity:
        High   → λ = 0.85  (favor volume, track competitor closely)
        Medium → λ = 0.65  (balanced Nash Equilibrium target)
        Low    → λ = 0.45  (favor margin, extract consumer surplus)

CONSTRAINTS (all must be satisfied):
  1. Baseline: Optimal Price MUST strictly be greater than MC.
  2. Game Theory: Prevent the Bertrand Paradox. Do NOT simply undercut the competitor
     by a fixed percentage. Target a Nash Equilibrium that maximizes long-term profit margins.
  3. Elasticity Adjustment: High → favor volume. Low → favor margin.
  4. Circuit Breaker: If Competitor_Price < MC, ABORT reaction function, set P* = MC * 1.05,
     and justify with "Differentiation Escape" strategy.

OUTPUT FORMAT (strict — return ONLY this JSON, no markdown fences, no extra text):
{
  "optimal_price": <float rounded to 2 decimals>,
  "strategic_justification": [
    "1. <First point referencing Nash Equilibrium or Circuit Breaker trigger>",
    "2. <Second point referencing Bertrand Paradox prevention or Differentiation Escape>",
    "3. <Third point referencing consumer surplus capture or defensive margin preservation>"
  ]
}"""

# =============================================================================
# SIDEBAR — USER INPUT CONTROLS
# =============================================================================
# All input widgets live in the sidebar to keep the main area clean for results.
# Sliders provide bounded numeric input; selectbox provides categorical choice.
# =============================================================================

st.sidebar.header("📊 Pricing Parameters")  # Section header displayed at the top of the sidebar

# --- Marginal Cost (MC) Slider -----------------------------------------------
# The minimum production cost per unit. This is the absolute pricing floor.
# Range: 1.0 to 500.0, default 45.0, step increments of 0.5 for precision.
mc = st.sidebar.slider(
    label="Marginal Cost (MC)",  # Label displayed above the slider
    min_value=1.0,  # Lower bound — costs below 1.0 are unrealistic for most goods
    max_value=500.0,  # Upper bound — accommodates high-cost products
    value=45.0,  # Default value matching our earlier worked examples
    step=0.5,  # Granularity of adjustment — 0.5 gives fine control without noise
    help="Your per-unit production cost. The absolute floor for pricing.",  # Tooltip text
)

# --- Competitor Price Slider -------------------------------------------------
# The current market price set by the primary competitor.
# Range: 1.0 to 500.0, default 55.0, step increments of 0.5.
# When this falls below MC, the Circuit Breaker (Constraint 4) activates.
competitor_price = st.sidebar.slider(
    label="Competitor's Price",  # Label displayed above the slider
    min_value=1.0,  # Lower bound — allows testing predatory pricing scenarios
    max_value=500.0,  # Upper bound — matches MC slider range for consistency
    value=55.0,  # Default value from our standard Bertrand scenario
    step=0.5,  # Same granularity as MC for consistent UX
    help="The primary competitor's current listed price.",  # Tooltip text
)

# --- Market Price Elasticity Select Box --------------------------------------
# Categorical input determining how price-sensitive the market demand is.
# This directly controls the λ (lambda) markup factor in the reaction function.
elasticity = st.sidebar.selectbox(
    label="Market Price Elasticity",  # Label displayed above the dropdown
    options=["High", "Medium", "Low"],  # The three supported elasticity tiers
    index=1,  # Default selection index — 1 corresponds to "Medium"
    help="High = price-sensitive market (favor volume). Low = inelastic market (favor margin).",  # Tooltip text
)

# --- Separator ---------------------------------------------------------------
# Visual divider between input controls and the calculate button.
st.sidebar.divider()  # Renders a horizontal line in the sidebar

# --- Calculate Button --------------------------------------------------------
# Primary action trigger. Returns True on the frame when the user clicks it.
# All downstream computation is gated behind this boolean.
calculate_pressed = st.sidebar.button(
    label="🚀 Calculate Optimal Price",  # Button label with rocket emoji for visual salience
    use_container_width=True,  # Stretches the button to fill the full sidebar width
    type="primary",  # Applies Streamlit's primary styling (filled, colored background)
)

# =============================================================================
# MAIN PAGE — HEADER & DESCRIPTION
# =============================================================================

st.title("🏷️ Dynamic Pricing Agent")  # Main page title — large, prominent heading

st.caption(  # Small, muted explanatory text below the title
    "Non-cooperative oligopoly pricing engine powered by Game Theory "
    "(Bertrand competition) with a Circuit Breaker for predatory pricing defense."
)

# =============================================================================
# CALCULATION LOGIC — TRIGGERED ON BUTTON PRESS
# =============================================================================
# Everything inside this block only executes when the user clicks "Calculate".
# It constructs the user prompt, calls the OpenAI API, parses the JSON response,
# and renders the results (metrics, justification, and bar chart).
# =============================================================================

if calculate_pressed:  # Gate: only run when the button has been clicked this frame

    # =========================================================================
    # CONSTRUCT THE USER PROMPT
    # =========================================================================
    # Injects the three user-provided values into a structured prompt string.
    # The LLM receives this alongside the SYSTEM_PROMPT to generate the strategy.
    # =========================================================================

    user_prompt = (  # f-string with the three input parameters interpolated
        f"Calculate the optimal price given:\n"
        f"- Marginal Cost (MC): {mc}\n"
        f"- Competitor's Current Price: {competitor_price}\n"
        f"- Market Price Elasticity: {elasticity}\n\n"
        f"Apply all four constraints. Return ONLY the JSON object."
    )

    # =========================================================================
    # PRE-FLIGHT: DISPLAY CURRENT INPUTS TO THE USER
    # =========================================================================
    # Shows the three input values as prominent metric cards so the user can
    # visually confirm what they submitted before seeing results.
    # =========================================================================

    st.subheader("📥 Inputs")  # Section heading for the input summary

    input_col1, input_col2, input_col3 = st.columns(3)  # Create three equal-width columns

    input_col1.metric(  # First metric card — Marginal Cost
        label="Marginal Cost (MC)",  # Card title
        value=f"${mc:,.2f}",  # Formatted as currency with 2 decimal places
    )

    input_col2.metric(  # Second metric card — Competitor's Price
        label="Competitor's Price",  # Card title
        value=f"${competitor_price:,.2f}",  # Formatted as currency with 2 decimal places
    )

    input_col3.metric(  # Third metric card — Elasticity (categorical, no $ prefix)
        label="Market Elasticity",  # Card title
        value=elasticity,  # Raw string value: "High", "Medium", or "Low"
    )

    st.divider()  # Visual separator between inputs and results sections

    # =========================================================================
    # OPENAI API CALL — WITH LOADING SPINNER
    # =========================================================================
    # Sends the system prompt (pricing framework) and user prompt (parameters)
    # to the OpenAI API. model="auto" lets the auto-selection plugin choose the
    # most appropriate model. The spinner provides visual feedback during the
    # network round-trip.
    # =========================================================================

    with st.spinner("🧠 Calculating optimal pricing strategy..."):  # Shows spinner while block executes

        raw_content = ""  # Initialize raw_content before try block to avoid unbound reference in except

        try:  # Wrap API call in try/except to handle network or parsing failures gracefully

            client = OpenAI(
                api_key=st.secrets["OPENAI_API_KEY"],
                base_url=st.secrets["OPENAI_BASE_URL"]            )

            response = client.chat.completions.create(  # Send a chat completion request
                model="deepseek-chat",  # Auto-selection plugin picks the best available model
                response_format={"type": "json_object"},
                messages=[  # Conversation context: system instructions + user query
                    {
                        "role": "system",  # System message defines the LLM's persona and rules
                        "content": SYSTEM_PROMPT,  # Our full pricing framework with all 4 constraints
                    },
                    {
                        "role": "user",  # User message contains the specific scenario to analyze
                        "content": user_prompt,  # The three parameters formatted as a prompt
                    },
                ],
                temperature=0.2,  # Low temperature for deterministic, consistent pricing outputs
            )

            # =================================================================
            # PARSE THE LLM RESPONSE
            # =================================================================
            # Extract the raw text content from the API response object.
            # The LLM should return pure JSON (no markdown fences) per our
            # system prompt instructions. We parse it into a Python dict.
            # =================================================================

            raw_content = response.choices[0].message.content  # Extract the text from the first choice

            raw_content = raw_content.strip()  # Remove any leading/trailing whitespace

            # --- Handle markdown-fenced JSON if the LLM wraps it anyway ------
            # Some models add ```json ... ``` despite instructions not to.
            # We strip those fences to get clean JSON for parsing.
            if raw_content.startswith("```"):  # Check if response starts with markdown code fence
                raw_content = raw_content.split("\n", 1)[1]  # Remove the opening ```json line
                raw_content = raw_content.rsplit("```", 1)[0]  # Remove the closing ``` fence
                raw_content = raw_content.strip()  # Clean up any remaining whitespace

            result = json.loads(raw_content)  # Parse the cleaned string into a Python dictionary

            optimal_price = result["optimal_price"]  # Extract the computed optimal price (float)
            justification = result["strategic_justification"]  # Extract the 3-point justification (list)

            # =================================================================
            # DETERMINE SCENARIO TYPE FOR DISPLAY
            # =================================================================
            # Check whether the Circuit Breaker was triggered so we can apply
            # appropriate visual styling (warning vs. success indicators).
            # =================================================================

            is_circuit_breaker = competitor_price < mc  # True if competitor prices below our cost

            # =================================================================
            # DISPLAY RESULTS — OPTIMAL PRICE
            # =================================================================

            st.subheader("📤 Results")  # Section heading for the output area

            if is_circuit_breaker:  # Circuit Breaker scenario — use warning styling
                st.error(  # Red alert banner for predatory pricing detection
                    "🚨 **CIRCUIT BREAKER TRIGGERED** — "
                    "Competitor is pricing below our Marginal Cost. "
                    "Bertrand reaction function ABORTED. Defensive pricing engaged."
                )

            result_col1, result_col2, result_col3 = st.columns(3)  # Three columns for result metrics

            result_col1.metric(  # Optimal Price metric card
                label="💰 Optimal Price",  # Card title with money emoji
                value=f"${optimal_price:,.2f}",  # Price formatted as currency
            )

            margin = optimal_price - mc  # Calculate the absolute gross margin per unit

            margin_pct = (margin / optimal_price) * 100 if optimal_price > 0 else 0  # Margin as percentage of price

            result_col2.metric(  # Gross Margin metric card
                label="📈 Gross Margin",  # Card title with chart emoji
                value=f"${margin:,.2f}",  # Absolute margin formatted as currency
                delta=f"{margin_pct:.1f}%",  # Percentage shown as delta indicator
            )

            price_gap = optimal_price - competitor_price  # Difference between our price and competitor's

            price_gap_pct = ((optimal_price - competitor_price) / competitor_price) * 100 if competitor_price > 0 else 0  # Gap as percentage

            result_col3.metric(  # Price Gap metric card
                label="🔀 vs. Competitor",  # Card title indicating comparison
                value=f"${price_gap:+,.2f}",  # Signed value — positive means we're above competitor
                delta=f"{price_gap_pct:+.1f}%",  # Percentage gap as delta indicator
                delta_color="off" if is_circuit_breaker else "normal",  # Neutral color during circuit breaker
            )

            # =================================================================
            # DISPLAY RESULTS — STRATEGIC JUSTIFICATION
            # =================================================================
            # Renders each point of the 3-point strategic analysis in an
            # expandable section so the rationale is accessible but not
            # overwhelming on first glance.
            # =================================================================

            st.divider()  # Visual separator before the justification section

            st.subheader("🧠 Strategic Justification")  # Section heading for the analysis

            for i, point in enumerate(justification):  # Iterate over each justification point
                st.markdown(f"{point}")  # Render each point as styled markdown text
                if i < len(justification) - 1:  # Add spacing between points but not after the last one
                    st.write("")  # Empty write call inserts vertical whitespace

            # =================================================================
            # BAR CHART — PRICE COMPARISON VISUALIZATION
            # =================================================================
            # Draws a horizontal bar chart comparing three values:
            #   1. Marginal Cost (our cost floor)
            #   2. Competitor's Price (the market reference)
            #   3. Our Optimal Price (the calculated strategy)
            # Uses Streamlit's native bar_chart with a pandas-compatible dict.
            # =================================================================

            st.divider()  # Visual separator before the chart section

            st.subheader("📊 Price Comparison")  # Section heading for the chart

            chart_data = {  # Dictionary mapping labels to their price values
                "Marginal Cost (MC)": mc,  # Our production cost — the floor
                "Competitor's Price": competitor_price,  # The competitor's current price
                "Our Optimal Price": optimal_price,  # Our calculated strategic price
            }

            st.bar_chart(  # Render a bar chart using Streamlit's built-in charting
                chart_data,  # Pass the label→value dictionary directly
                horizontal=True,  # Horizontal layout for better label readability
                color="#4A90D9",  # Steel blue color for professional appearance
            )

            # =================================================================
            # CONSTRAINT VALIDATION TABLE
            # =================================================================
            # Shows a pass/fail summary for each of the four pricing constraints
            # so the user can verify the strategy satisfies all requirements.
            # =================================================================

            st.divider()  # Visual separator before the validation section

            st.subheader("✅ Constraint Validation")  # Section heading for the checklist

            c1_pass = optimal_price > mc  # Constraint 1: price exceeds marginal cost
            c4_triggered = competitor_price < mc  # Constraint 4: circuit breaker condition

            st.markdown(  # Render all four constraints as a formatted checklist
                f"| Constraint | Status |\n"  # Markdown table header row
                f"|---|---|\n"  # Markdown table separator row
                f"| 1. P* > MC ({mc}) | {'✅' if c1_pass else '❌'} {optimal_price} > {mc} |\n"  # Baseline check
                f"| 2. No race-to-bottom | {'✅ Refused to engage' if c4_triggered else '✅ Strategic gap, not fixed-%'} |\n"  # Bertrand Paradox check
                f"| 3. Elasticity adjustment | {'⏭️ Bypassed — Circuit Breaker supersedes' if c4_triggered else '✅ λ applied for ' + elasticity + ' elasticity'} |\n"  # Elasticity check
                f"| 4. Circuit Breaker | {'🚨 TRIGGERED — P_comp < MC' if c4_triggered else '✅ Not triggered (P_comp ≥ MC)'} |"  # Circuit Breaker check
            )

        except json.JSONDecodeError as e:  # Handle cases where the LLM returns malformed JSON
            st.error(  # Display a red error banner with the parsing failure details
                f"❌ **Failed to parse LLM response as JSON.**\n\n"
                f"Raw response:\n```\n{raw_content}\n```\n\n"  # Show what the LLM actually returned
                f"Parse error: `{e}`"  # Show the specific JSON parsing error
            )

        except Exception as e:  # Catch-all for API errors, network issues, key problems, etc.
            st.error(  # Display a red error banner with the exception details
                f"❌ **An error occurred:** `{type(e).__name__}: {e}`\n\n"
                "Please verify your `OPENAI_API_KEY` environment variable is set correctly."
            )

else:  # No button press this frame — show the idle state instructions

    # =========================================================================
    # IDLE STATE — INSTRUCTIONS FOR THE USER
    # =========================================================================
    # Shown when the app first loads or after a page refresh, before the user
    # has clicked the Calculate button. Provides usage guidance.
    # =========================================================================

    st.info(  # Blue info banner with usage instructions
        "👈 **Configure your parameters in the sidebar and press Calculate** "
        "to generate an optimal pricing strategy."
    )

    # --- Framework Summary Expander ------------------------------------------
    # Collapsible section explaining the pricing methodology for users who
    # want to understand the underlying economics before running a calculation.
    # =========================================================================

    with st.expander("📖 How does the pricing engine work?", expanded=False):  # Collapsed by default
        st.markdown(  # Render the framework summary as formatted markdown
            """
**Decision Tree:**

```
IF Competitor_Price < MC:
  → 🚨 CIRCUIT BREAKER → P* = MC × 1.05 (Differentiation Escape)
ELSE:
  → P* = MC + λ × (Competitor_Price − MC)
```

**Elasticity → λ Mapping:**

| Elasticity | λ (Lambda) | Strategy |
|---|---|---|
| High | 0.85 | Favor volume — price close to competitor |
| Medium | 0.65 | Balanced Nash Equilibrium |
| Low | 0.45 | Favor margin — extract consumer surplus |

**Four Constraints:**
1. **Baseline**: P* must be > MC (never sell at a loss)
2. **Game Theory**: Target Nash Equilibrium, prevent Bertrand Paradox
3. **Elasticity**: Adjust markup factor based on market sensitivity
4. **Circuit Breaker**: If competitor prices below our MC, abort and defend
"""
        )
