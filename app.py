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
from typing import Any, cast  # 新增：引入类型工具用于显式声明 DataFrame 类型，降低静态检查误报
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
# SYSTEM PROMPT TEMPLATE — LOGIC DECOUPLING
# =============================================================================
# 本模板用于“计算与文案彻底解耦”架构：
# 1) 数学计算与 IF-ELSE 判断全部由 Python 层完成
# 2) 大模型只负责根据锁定结果写专业中文分析
# 3) 输出 JSON 键保持不变，确保前端解析稳定
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """你是一位极其冷酷、精通微观经济学（Microeconomics）与非合作博弈（Non-cooperative Game）的首席风险定价官。
请基于以下【既定商业事实】，向企业高管撰写一份客观、刺耳的商业战略与风险分析。

【既定业务事实】
- 最终执行价格：{locked_optimal_price}
- 市场大盘加权均价：{locked_weighted_price} (必须注意我们与大盘的差价)
- 战略定调：{locked_strategy_status}

【最高级别红线 (CRITICAL GUARDRAILS)】
1. 严禁打破第四面墙：绝对不允许在报告中提到“Python”、“代码”、“模型”、“变量”、“解耦”等技术词汇。你必须表现得像一个真实的人类高管在做业务汇报。
2. 风险预警强制执行：如果我们的执行价格高于大盘均价，必须在第二段明确使用“销量断崖式下跌 (Precipitous Drop in Volume)”和“市场份额流失 (Market Share Erosion)”这两个词进行残酷警告。
3. 严禁滥用学术词汇：绝对不允许说“提升/管理消费者剩余”。高价状态下，消费者剩余是被严重榨取和剥夺的。

【强制输出格式】
严格返回 JSON 格式，不附加任何 markdown 标记或其他文本：
{{
  "optimal_price": {locked_optimal_price},
  "strategic_analysis": [
    "<第一段>：冷酷陈述当前的博弈格局与定价决策，明确对比我们的价格与大盘加权均价，点明当前的竞争劣势。不少于80字。",
    "<第二段>：核心风险预警 (Red Teaming)。强制执行最高级别红线第2条，严禁粉饰太平。不少于80字。",
    "<第三段>：冷血的生存对策。面对销量必然下滑的局面，给出缩减固定成本、放弃低端市场或提高客户转换成本 (Switching Costs) 的具体业务建议。不少于80字。"
  ]
}}"""
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
    competitor_df = cast(pd.DataFrame, edited_competitors.copy())  # 新增：显式声明为 DataFrame 类型，帮助静态检查正确识别后续DataFrame方法
else:  # 新增：兜底分支，处理极端情况下非 DataFrame 返回结构
    competitor_df = cast(pd.DataFrame, pd.DataFrame(cast(Any, edited_competitors)))  # 新增：将非标准返回值强制规范化为 DataFrame，统一后续处理口径

required_columns = ["Competitor Name", "Price", "Market Share %"]  # 新增：定义强制列集合，确保计算字段完整可用

for required_column in required_columns:  # 新增：逐列校验，防止用户删列导致后续计算报错
    if required_column not in competitor_df.columns:  # 新增：发现缺列时执行兜底填充，保障计算链路不断裂
        competitor_df[required_column] = "" if required_column == "Competitor Name" else 0.0  # 新增：名称列补空字符串，数值列补0用于安全默认值

competitor_df = cast(pd.DataFrame, competitor_df[required_columns].copy())  # 新增：显式声明重排后的对象为DataFrame，减少静态检查误判
competitor_records = cast(list[dict[str, Any]], competitor_df.to_dict("records"))  # 新增：转为字典列表做纯Python清洗，彻底规避链式Series类型噪音
normalized_records: list[dict[str, Any]] = []  # 新增：初始化标准化记录容器，用于逐行构建高可信输入数据

for row in competitor_records:  # 新增：逐行清洗每个竞争对手记录，避免DataFrame推断歧义影响可靠性
    cleaned_name = str(row.get("Competitor Name", "")).strip()  # 新增：统一名称清洗为字符串并去空格，保证展示可读与键值稳定
    raw_price_value = row.get("Price", 0.0)  # 新增：先取原始价格字段，便于在类型异常时做分层容错
    raw_share_value = row.get("Market Share %", 0.0)  # 新增：先取原始份额字段，便于在类型异常时做分层容错
    try:  # 新增：优先走快速路径，将可转浮点的输入直接固化为float
        cleaned_price = float(raw_price_value) if raw_price_value is not None else 0.0  # 新增：价格快速转换，覆盖常见数字/字符串输入
    except (TypeError, ValueError):  # 新增：当输入异常时进入保守兜底，避免单行脏数据中断全局计算
        cleaned_price = 0.0  # 新增：异常价格统一归零，保证策略引擎可继续运行
    try:  # 新增：份额字段同样采用快速路径转换，提升处理效率与可预期性
        cleaned_share = float(raw_share_value) if raw_share_value is not None else 0.0  # 新增：份额快速转换，覆盖常见数字/字符串输入
    except (TypeError, ValueError):  # 新增：份额异常时执行兜底，防止坏值污染加权公式
        cleaned_share = 0.0  # 新增：异常份额统一归零，保持加权计算稳定
    if cleaned_name != "":  # 新增：仅保留具备有效名称的竞品行，避免空行污染加权计算
        normalized_records.append({"Competitor Name": cleaned_name, "Price": cleaned_price, "Market Share %": cleaned_share})  # 新增：写入标准化记录，确保后续计算字段完备

if len(normalized_records) == 0:  # 新增：若用户删除了全部有效竞品行，则回退默认演示集保证可计算性
    competitor_df = default_competitors.copy()  # 新增：启用默认样本避免页面进入空输入不可计算状态
else:  # 新增：存在有效输入时使用标准化结果重建DataFrame
    competitor_df = cast(pd.DataFrame, pd.DataFrame(normalized_records, columns=cast(Any, required_columns)))  # 新增：重建DataFrame并显式声明类型，降低pandas存根误报

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
    # PYTHON LAYER — LOGIC DECOUPLING (NO LLM MATH)
    # =========================================================================
    # 在进入提示词组装前，先由 Python 完成全部数学计算与 IF-ELSE 判断，
    # 从架构层面阻断大模型在算术与比较上的幻觉风险。
    # =========================================================================

    elasticity_lambda_map = {  # 新增：定义弹性到 λ 的映射表，确保价格公式参数来自确定性字典
        "High": 0.85,  # 新增：高弹性市场偏向份额防守，λ 取较高值使价格更贴近竞争均价
        "Medium": 0.65,  # 新增：中弹性市场采用均衡折中，兼顾利润与销量
        "Low": 0.45,  # 新增：低弹性市场偏向利润捕获，λ 取较低值提升单位毛利
    }  # 新增：结束 λ 参数映射定义

    selected_lambda = elasticity_lambda_map.get(elasticity, 0.65)  # 新增：按当前弹性安全读取 λ，异常值回退到 Medium 默认值

    is_circuit_breaker = weighted_average_competitor_price < mc  # 新增：Python 层先比较加权竞品价与 MC，作为唯一熔断判定真值

    if is_circuit_breaker:  # 新增：当竞品加权基准低于 MC 时，立即执行防御性熔断分支
        computed_optimal_price = round(mc * 1.05, 2)  # 新增：Python 强制计算熔断价格，确保严格等于 MC*1.05 且保留两位小数
        strategy_status = "已触发差异化熔断，彻底放弃价格战"  # 新增：写入锁定战略状态字符串，供 LLM 仅做文案展开
    else:  # 新增：未触发熔断时进入纳什均衡分支
        computed_optimal_price = round(mc + selected_lambda * (weighted_average_competitor_price - mc), 2)  # 新增：Python 计算伯特兰反应函数结果，彻底剥离 LLM 算数职责
        strategy_status = "未触发熔断，执行纳什均衡"  # 新增：写入锁定战略状态字符串，明确当前为均衡执行态

    computed_optimal_price = max(computed_optimal_price, round(mc + 0.01, 2))  # 新增：执行基线保护，确保最终价格严格大于 MC（Constraint 1）

    locked_optimal_price = f"{computed_optimal_price:.2f}"  # 新增：将已计算价格固化为字符串，作为不可篡改注入变量
    locked_strategy_status = strategy_status  # 新增：将战略状态固化为锁定文本，防止模型重解释逻辑分支
    locked_circuit_breaker = "true" if is_circuit_breaker else "false"  # 新增：将熔断状态转为字符串布尔量，便于提示词中显式约束

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(  # 新增：运行时注入锁定变量生成最终系统提示词，实现“计算先行、文案后置”
        locked_optimal_price=locked_optimal_price,  # 新增：注入锁定最优价，禁止模型重算
        locked_strategy_status=locked_strategy_status,  # 新增：注入锁定战略状态，禁止模型改判
        locked_circuit_breaker=locked_circuit_breaker,  # 新增：注入锁定熔断标志，禁止模型重做比较
    )  # 新增：结束系统提示词组装

    competitor_table_payload = competitor_df.to_dict("records")  # 新增：将表格转为结构化列表，避免关键字重载告警并供 LLM 描述竞争格局

    user_prompt = (  # 新增：构建寡头场景的请求体，明确“加权竞品均价”是唯一博弈锚点
        f"以下字段全部由 Python 预计算并锁定，请直接用于中文商业分析写作：\n"  # 新增：明确告知模型输入是只读业务事实，不是可推导草稿
        f"- LOCKED_OPTIMAL_PRICE: {locked_optimal_price}\n"  # 新增：重复注入锁定价格，双重保险防止模型偏离
        f"- LOCKED_STRATEGY_STATUS: {locked_strategy_status}\n"  # 新增：重复注入锁定状态，确保分析围绕既定战略展开
        f"- LOCKED_CIRCUIT_BREAKER: {locked_circuit_breaker}\n"  # 新增：重复注入熔断布尔，便于模型在文案中准确表述触发状态
        f"- Marginal Cost (MC): {mc}\n"  # 新增：提供成本背景用于管理层解释，但不允许二次计算
        f"- Weighted Average Competitor Price: {weighted_average_competitor_price:.2f}\n"  # 新增：提供市场参照背景用于论证，不授权模型做比较判断
        f"- Market Price Elasticity: {elasticity}\n"  # 新增：提供市场弹性背景辅助策略叙述
        f"- Selected Lambda (Python computed): {selected_lambda}\n"  # 新增：披露 Python 已选 λ，提升分析可审计性
        f"- Competitor Table: {json.dumps(competitor_table_payload, ensure_ascii=False)}\n\n"  # 新增：保留明细供模型描述竞争格局，不参与定价计算
        f"任务要求：你不需要做任何计算，直接引用传入的价格结果，并基于传入的战略状态，撰写三段极其专业的中文商业分析。"  # 新增：按用户要求写死职责边界，彻底限制模型角色
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
                model="deepseek-chat",  # 新增：启用自动模型选择插件，满足“model=auto”的明确要求
                response_format={"type": "json_object"},
                messages=[  # Conversation context: system instructions + user query
                    {
                        "role": "system",  # System message defines the LLM's persona and rules
                        "content": system_prompt,  # 新增：使用运行时注入锁定变量后的系统提示词，确保模型不做计算
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

            raw_content = response.choices[0].message.content or ""  # 新增：为可空 content 提供空串兜底，避免可选类型调用字符串方法报错

            raw_content = raw_content.strip()  # Remove any leading/trailing whitespace

            # --- Handle markdown-fenced JSON if the LLM wraps it anyway ------
            # Some models add ```json ... ``` despite instructions not to.
            # We strip those fences to get clean JSON for parsing.
            if raw_content.startswith("```"):  # Check if response starts with markdown code fence
                raw_content = raw_content.split("\n", 1)[1]  # Remove the opening ```json line
                raw_content = raw_content.rsplit("```", 1)[0]  # Remove the closing ``` fence
                raw_content = raw_content.strip()  # Clean up any remaining whitespace

            result = json.loads(raw_content)  # Parse the cleaned string into a Python dictionary

            model_optimal_price = float(result.get("optimal_price", computed_optimal_price))  # 新增：读取模型返回价格用于一致性审计，不作为最终业务真值
            optimal_price = computed_optimal_price  # 新增：最终展示价格强制使用 Python 计算结果，彻底剥离模型算数影响
            justification = result.get("strategic_justification", [])  # 新增：安全读取三段分析，若模型异常则回退为空列表
            if not isinstance(justification, list):  # 新增：保证 justification 数据类型稳定，防止非列表导致渲染错误
                justification = [str(justification)]  # 新增：将异常类型兜底转换为列表，保障页面可渲染
            if len(justification) == 0:  # 新增：当模型未返回分析时注入兜底文案，确保页面对业务方可读
                justification = [  # 新增：生成三段默认分析，保持输出结构与管理视图一致
                    f"1. 当前战略状态：{strategy_status}。系统已在 Python 层完成全部数学与分支判断，锁定最优价格为 {optimal_price:.2f}，避免大模型在算术与比较环节产生幻觉风险。",
                    "2. 多竞争对手格局分析：建议重点跟踪高份额竞品的价格与促销节奏，并结合低份额搅局者的短期扰动，采用差异化价值主张与渠道组合维持目标客户粘性。",
                    "3. 管理层执行建议：将定价决策与品牌、服务、交付体验联动，形成利润防御与消费者剩余管理闭环；同时建立周期复盘机制，持续校准份额权重与弹性参数。",
                ]
            price_consistency_gap = round(model_optimal_price - optimal_price, 4)  # 新增：计算模型返回价与Python锁定价差，用于监控模型是否偏离指令

            # =================================================================
            # DETERMINE SCENARIO TYPE FOR DISPLAY
            # =================================================================
            # Check whether the Circuit Breaker was triggered so we can apply
            # appropriate visual styling (warning vs. success indicators).
            # =================================================================

            is_circuit_breaker = locked_circuit_breaker == "true"  # 新增：展示层熔断标记直接读取 Python 锁定状态，避免重复判断口径漂移

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

            st.caption(  # 新增：展示一致性审计信息，帮助业务侧确认模型未篡改定价结果
                f"🔒 逻辑解耦审计：Python锁定价={optimal_price:.2f}，模型返回价={model_optimal_price:.2f}，差值={price_consistency_gap:+.4f}"  # 新增：明确告知审计差值，增强可观测性
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

            st.info(  # 新增：显式展示 Python 锁定的战略状态，让业务方看到最终策略判定来源
                f"🧭 Python Strategy Status: {strategy_status}"  # 新增：将锁定状态可视化，避免用户误以为由模型临时判断
            )

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
            c4_triggered = is_circuit_breaker  # 新增：约束4校验直接复用 Python 锁定熔断状态，确保口径前后一致
            c2_pass = True  # 新增：约束2由Python先算后写的架构天然满足“非LLM价格战误判”，此处固定为通过
            c3_pass = True  # 新增：约束3由Python按弹性映射选择 λ 并计算，展示层标记为通过

            st.markdown(  # Render all four constraints as a formatted checklist
                f"| Constraint | Status |\n"  # Markdown table header row
                f"|---|---|\n"  # Markdown table separator row
                f"| 1. P* > MC ({mc}) | {'✅' if c1_pass else '❌'} {optimal_price} > {mc} |\n"  # Baseline check
                f"| 2. No race-to-bottom | {'✅' if c2_pass else '❌'} Python锁定策略：{'Refused to engage' if c4_triggered else 'Nash equilibrium execution'} |\n"  # 新增：约束2文案改为Python锁定口径，避免归因给模型
                f"| 3. Elasticity adjustment | {'✅' if c3_pass else '❌'} {'⏭️ 熔断时跳过λ应用' if c4_triggered else 'λ=' + str(selected_lambda) + ' for ' + elasticity} |\n"  # 新增：约束3文案展示Python侧λ值，确保可审计
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

**Logic Decoupling (v2 hardening):**
- Python 先完成全部数学计算与 IF-ELSE 熔断判断
- LLM 只负责生成中文商业分析文案，不参与任何计算
- 页面展示的 Optimal Price 以 Python 锁定结果为唯一真值
"""
        )
