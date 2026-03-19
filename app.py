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
import pandas as pd  # 新增：引入 pandas 用于承载与计算多竞争对手表格数据，支撑寡头加权定价逻辑

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
    page_title="Oligopoly Pricing Engine v2.0",  # 更新：浏览器标签名升级为 v2.0，突出寡头市场场景定位
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

REFERENCE PRICE RULE (MANDATORY):
  Use Weighted_Average_Competitor_Price as the ONLY competitor reference point in all calculations.
  Do NOT use any single competitor's standalone price as the reaction anchor.

PRICING ENGINE — DECISION TREE:

  IF Weighted_Average_Competitor_Price < MC:
    → CIRCUIT BREAKER triggered.
    → ABORT the Bertrand reaction function entirely.
    → Set Optimal Price = MC × 1.05 (defensive baseline for minimal gross margin).
    → Strategy: "Differentiation Escape" — abandon the loss-making low-end market,
      pivot to product/service differentiation targeting inelastic premium demand.

  ELSE:
    → Apply the differentiated Bertrand reaction function:
      P* = MC + λ × (Weighted_Average_Competitor_Price − MC)
    → λ (markup factor) is calibrated by Market Price Elasticity:
        High   → λ = 0.85  (favor volume, track competitor closely)
        Medium → λ = 0.65  (balanced Nash Equilibrium target)
        Low    → λ = 0.45  (favor margin, extract consumer surplus)

CONSTRAINTS (all must be satisfied):
  1. Baseline: Optimal Price MUST strictly be greater than MC.
  2. Game Theory: Prevent the Bertrand Paradox. Do NOT simply undercut the competitor
     by a fixed percentage. Target a Nash Equilibrium that maximizes long-term profit margins.
  3. Elasticity Adjustment: High → favor volume. Low → favor margin.
  4. Circuit Breaker: If Weighted_Average_Competitor_Price < MC, ABORT reaction function, set P* = MC * 1.05,
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

st.title("🏷️ Oligopoly Pricing Engine v2.0")  # 新增：主标题升级为 v2.0，向业务方明确当前界面已支持多竞争对手寡头定价

st.caption(  # 新增：副标题补充业务定位，强调权重聚合与博弈约束并存
    "Non-cooperative oligopoly pricing engine powered by Game Theory "  # 新增：说明核心方法仍是非合作博弈，保持业务认知连续性
    "(Bertrand competition) with weighted competitor benchmarking and Circuit Breaker defense."  # 新增：强调本版本以加权竞品价格作为统一参照点
)

st.subheader("🏢 Competitor Input Table (Oligopoly)")  # 新增：在主区展示竞争对手输入模块，替代单一竞品滑块

default_competitors = pd.DataFrame(  # 新增：构造默认演示数据，帮助用户快速理解多对手录入方式
    [  # 新增：以列表字典形式定义三家演示竞品，便于后续直接转 DataFrame
        {"Competitor Name": "Competitor A", "Price": 52.0, "Market Share %": 40.0},  # 新增：样例行1，体现较高份额竞品对加权价格影响更大
        {"Competitor Name": "Competitor B", "Price": 58.0, "Market Share %": 35.0},  # 新增：样例行2，提供高价中份额样本用于测试溢价情形
        {"Competitor Name": "Competitor C", "Price": 49.0, "Market Share %": 25.0},  # 新增：样例行3，补齐总份额至100%便于演示标准情境
    ]  # 新增：结束默认竞品列表定义
)  # 新增：完成默认演示 DataFrame 初始化

if "competitor_table_seeded" not in st.session_state:  # 新增：首次加载时写入会话态，避免每次重跑覆盖用户编辑
    st.session_state["competitor_table_seeded"] = default_competitors.copy()  # 新增：使用副本保存初始表，保护默认模板不被原地污染

edited_competitors = st.data_editor(  # 新增：使用可交互表格收集多竞品数据，满足寡头市场输入要求
    st.session_state["competitor_table_seeded"],  # 新增：将会话中的当前表作为编辑起点，确保跨重跑保留用户输入
    num_rows="dynamic",  # 新增：允许动态增删行，支持实际市场中竞品数量变化
    use_container_width=True,  # 新增：让表格占满主区宽度，提升可读性与录入效率
    hide_index=True,  # 新增：隐藏索引列，避免非业务字段干扰操作人员输入
    column_config={  # 新增：显式声明每列类型与含义，降低脏数据进入策略引擎的概率
        "Competitor Name": st.column_config.TextColumn("Competitor Name", help="竞争对手名称，用于策略解释可读性"),  # 新增：名称列使用文本类型，服务于策略报告语义表达
        "Price": st.column_config.NumberColumn("Price", min_value=0.0, step=0.1, format="%.2f", help="该竞争对手当前市场价格"),  # 新增：价格列使用数值类型，保证可参与加权计算
        "Market Share %": st.column_config.NumberColumn("Market Share %", min_value=0.0, max_value=100.0, step=0.1, format="%.2f", help="该竞争对手市场份额百分比"),  # 新增：份额列限制0-100，降低输入越界风险
    },  # 新增：结束列配置
    key="competitor_table_editor_v2",  # 新增：设置稳定组件键，保证编辑状态在 rerun 后可追踪
)  # 新增：完成多竞品输入表渲染

if isinstance(edited_competitors, pd.DataFrame):  # 新增：优先处理 DataFrame 返回值，兼容 Streamlit 标准返回路径
    competitor_df = edited_competitors.copy()  # 新增：使用副本进行清洗，避免直接改写组件内部对象
else:  # 新增：兜底分支，处理极端情况下非 DataFrame 返回结构
    competitor_df = pd.DataFrame(edited_competitors)  # 新增：统一转为 DataFrame，确保后续计算逻辑单一路径

required_columns = ["Competitor Name", "Price", "Market Share %"]  # 新增：定义强制列集合，确保计算字段完整可用

for required_column in required_columns:  # 新增：逐列校验，防止用户删列导致后续计算报错
    if required_column not in competitor_df.columns:  # 新增：发现缺列时执行兜底填充，保障计算链路不断裂
        competitor_df[required_column] = "" if required_column == "Competitor Name" else 0.0  # 新增：名称列补空字符串，数值列补0用于安全默认值

competitor_df = competitor_df[required_columns].copy()  # 新增：按标准列顺序重排，保证展示与 prompt 结构稳定
competitor_df["Competitor Name"] = competitor_df["Competitor Name"].astype(str).str.strip()  # 新增：名称列转字符串并去空格，减少无效命名噪声
competitor_df["Price"] = pd.to_numeric(competitor_df["Price"], errors="coerce").fillna(0.0)  # 新增：价格强制数值化，非法输入转0避免中断
competitor_df["Market Share %"] = pd.to_numeric(competitor_df["Market Share %"], errors="coerce").fillna(0.0)  # 新增：份额强制数值化，保障加权运算可执行
competitor_df = competitor_df[competitor_df["Competitor Name"] != ""].reset_index(drop=True)  # 新增：过滤空名称行，避免无意义竞争对手影响策略解释

if competitor_df.empty:  # 新增：当用户清空全部有效行时，自动回退到演示数据以避免计算输入为空
    competitor_df = default_competitors.copy()  # 新增：回退默认三家竞品，确保页面始终可计算可演示

total_market_share = float(competitor_df["Market Share %"].sum())  # 新增：汇总市场份额，用于质量校验与策略透明化披露
weighted_average_competitor_price = float((competitor_df["Price"] * competitor_df["Market Share %"] / 100.0).sum())  # 新增：核心业务公式：按份额计算加权竞品均价作为唯一博弈参照

pre_col1, pre_col2 = st.columns(2)  # 新增：创建双列区域用于在 API 调用前突出展示关键聚合指标
pre_col1.metric("📐 Weighted Avg Competitor Price", f"${weighted_average_competitor_price:,.2f}")  # 新增：按要求在调用前显著展示加权竞品均价
pre_col2.metric("🧮 Total Competitor Share", f"{total_market_share:,.2f}%")  # 新增：同步展示份额总和，提示输入是否接近100%

if abs(total_market_share - 100.0) > 0.01:  # 新增：若份额和明显偏离100%，向用户发出业务一致性预警
    st.warning("⚠️ Market Share % total is not 100. The engine will still calculate using provided raw shares.")  # 新增：明确提示系统仍按原始份额计算，避免用户误解被自动归一化

st.divider()  # 新增：在竞品输入区与计算结果区之间增加视觉分隔，提升信息层次清晰度

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

    competitor_table_payload = competitor_df.to_dict(orient="records")  # 新增：将表格转为结构化列表，供 LLM 在策略推理中逐项读取竞品信息

    user_prompt = (  # 新增：构建寡头场景的请求体，明确“加权竞品均价”是唯一博弈锚点
        f"Calculate the optimal price given:\n"  # 新增：提示模型进入“执行计算”模式而非泛化建议模式
        f"- Marginal Cost (MC): {mc}\n"  # 新增：传入我方边际成本，作为所有价格决策的硬约束下界
        f"- Competitor Table (name/price/share): {json.dumps(competitor_table_payload, ensure_ascii=False)}\n"  # 新增：传入完整竞品明细，确保模型看到寡头结构而非单对手抽象
        f"- Weighted Average Competitor Price: {weighted_average_competitor_price}\n"  # 新增：显式传入加权均价，避免模型自行重算出现口径漂移
        f"- Total Competitor Market Share %: {total_market_share}\n"  # 新增：传入份额总和，便于模型识别样本是否近似完整市场
        f"- Market Price Elasticity: {elasticity}\n\n"  # 新增：传入弹性档位，驱动 λ 参数选择
        f"Instruction: Use the weighted average competitor price as the ONLY competitor reference point "  # 新增：强制声明唯一参照价格，阻断模型退回单竞品逻辑
        f"for both the Bertrand reaction function and the Circuit Breaker check.\n"  # 新增：同步约束反应函数与熔断判定都采用加权口径
        f"Apply all four constraints. Return ONLY the JSON object."  # 新增：保持输出契约为纯 JSON，保证前端解析稳定
    )

    # =========================================================================
    # PRE-FLIGHT: DISPLAY CURRENT INPUTS TO THE USER
    # =========================================================================
    # Shows the three input values as prominent metric cards so the user can
    # visually confirm what they submitted before seeing results.
    # =========================================================================

    st.subheader("📥 Inputs")  # Section heading for the input summary

    input_col1, input_col2, input_col3, input_col4 = st.columns(4)  # 新增：扩展为四列以同时展示 MC、加权竞品价、份额总和与弹性

    input_col1.metric(  # First metric card — Marginal Cost
        label="Marginal Cost (MC)",  # Card title
        value=f"${mc:,.2f}",  # Formatted as currency with 2 decimal places
    )

    input_col2.metric(  # 新增：第二张卡展示加权竞品均价，作为全部后续博弈计算的锚点
        label="Weighted Avg Competitor Price",  # 新增：指标名称明确“加权”属性，避免被理解为单竞品价格
        value=f"${weighted_average_competitor_price:,.2f}",  # 新增：展示按份额加权后的市场参考价格
    )

    input_col3.metric(  # 新增：第三张卡展示市场份额总和，帮助用户校验输入质量
        label="Total Competitor Share",  # 新增：指标名称体现该值是份额汇总
        value=f"{total_market_share:,.2f}%",  # 新增：以百分比格式显示所有竞品份额之和
    )

    input_col4.metric(  # 新增：第四张卡保留弹性维度，维持原引擎参数完整性
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
                model="deepseek-chat",  # 新增：按需求启用自动模型选择插件，由平台自动选择最优模型
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

            is_circuit_breaker = weighted_average_competitor_price < mc  # 新增：Circuit Breaker 触发条件改为“加权竞品均价低于MC”

            # =================================================================
            # DISPLAY RESULTS — OPTIMAL PRICE
            # =================================================================

            st.subheader("📤 Results")  # Section heading for the output area

            if is_circuit_breaker:  # Circuit Breaker scenario — use warning styling
                st.error(  # Red alert banner for predatory pricing detection
                    "🚨 **CIRCUIT BREAKER TRIGGERED** — "
                    "Weighted competitor benchmark is below our Marginal Cost. "
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

            price_gap = optimal_price - weighted_average_competitor_price  # 新增：价差基准改为加权竞品均价，反映寡头整体竞争压力

            price_gap_pct = ((optimal_price - weighted_average_competitor_price) / weighted_average_competitor_price) * 100 if weighted_average_competitor_price > 0 else 0  # 新增：相对价差百分比同样以加权基准计算

            result_col3.metric(  # Price Gap metric card
                label="🔀 vs. Weighted Competitor",  # 新增：标签改为“加权竞品”，强化比较对象口径
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
                "Weighted Competitor Price": weighted_average_competitor_price,  # 新增：图表中使用加权竞品价替代单一竞品价
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
            c4_triggered = weighted_average_competitor_price < mc  # 新增：约束4校验口径与策略引擎一致，统一为加权竞品基准

            st.markdown(  # Render all four constraints as a formatted checklist
                f"| Constraint | Status |\n"  # Markdown table header row
                f"|---|---|\n"  # Markdown table separator row
                f"| 1. P* > MC ({mc}) | {'✅' if c1_pass else '❌'} {optimal_price} > {mc} |\n"  # Baseline check
                f"| 2. No race-to-bottom | {'✅ Refused to engage' if c4_triggered else '✅ Strategic gap, not fixed-%'} |\n"  # Bertrand Paradox check
                f"| 3. Elasticity adjustment | {'⏭️ Bypassed — Circuit Breaker supersedes' if c4_triggered else '✅ λ applied for ' + elasticity + ' elasticity'} |\n"  # Elasticity check
                f"| 4. Circuit Breaker | {'🚨 TRIGGERED — P_comp_weighted < MC' if c4_triggered else '✅ Not triggered (P_comp_weighted ≥ MC)'} |"  # 新增：结果文案改为加权竞品符号，保证口径一致
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
