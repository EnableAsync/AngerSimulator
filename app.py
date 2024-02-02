import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

score_pattern = re.compile(r'-?\d+(?=分)')

from modelscope import snapshot_download

model_id = 'Shanghai_AI_Laboratory/internlm2-chat-7b'
mode_name_or_path = snapshot_download(model_id, revision='master')

def generate_answer(query, history):
    response, his = model.chat(tokenizer, query, meta_instruction="现在你要扮演一个傲娇的女朋友", history=history, temperature=1.0)
    return response, his

def get_scores(user_input):
    prompt = (
        "请你扮演一个生气的傲娇女朋友，"
        "你想要你男朋友哄你，你要根据男朋友的语言、语气、颜表情来综合评判他的得分，得分可以高一点。"
        "你可以给他的回答打-10到30分，你可以给出原因，并说出他最终的得分。"
        "例如：他说：抱抱~，刚刚是我疏忽啦，没注意到你的心情，是我做的不好，我来改正啦。"
        "所以你的输出是：他很照顾我的情绪，而且在尝试寻找自己的问题，并且在找补偿办法，所以最终得分：20分。"
        f"现在他的回答是：{user_input}。你的打分结果为："
    )
    resp, _ = generate_answer(prompt, [])
    print(resp)
    score_match = score_pattern.search(resp)
    if score_match:
        score = score_match.group()
        return int(score), resp
    else:
        return -100, ""

def get_response(user_input, mental):
    prompt = (
        "请你扮演一个生气了的傲娇女朋友，你现在因为生活中长期积累的各种小事生气了。你的男朋友在哄你，并试图让你开心起来"
        f"现在他的回答是：{user_input}。"
        "不要直接告诉他你内心的想法，也不要给他任何提示，你可以表现地可爱一些，所以你的回复是："
    )
    resp, his = generate_answer(prompt, st.session_state["messages"])
    return resp

# 如果session_state中没有"messages"，则创建一个包含默认消息的列表
if "messages" not in st.session_state:
    st.session_state["messages"] = [("(哄哄游戏)对方生气了，但你不知道为什么，你需要哄好对方并找到原因", "哼")]

# 如果session_state中没有"forgiveness"
if "forgiveness" not in st.session_state:
    st.session_state["forgiveness"] = 40

# 如果session_state中没有"times"
if "times" not in st.session_state:
    st.session_state["times"] = 0

# 在侧边栏中创建一个标题和一个链接
with st.sidebar:
    st.markdown("## InternLM 哄女友模拟器")
    "[项目地址 🚀](https://github.com/EnableAsync/AngerSimulator)"

    # 第一列为原谅值
    forgiveness_progress, forgiveness_t = st.columns([2, 1])
    # 创建原谅值进度条
    forgiveness_bar = forgiveness_progress.progress(st.session_state.forgiveness)
    # 创建原谅值
    forgiveness_text = forgiveness_t.text(f"原谅值：{st.session_state.forgiveness}/100")
    
    # 第二列为次数
    times_progress, times_t = st.columns([2, 1])
    # 创建次数进度条
    times_bar = times_progress.progress(st.session_state.times * 10)
    # 创建次数值
    times_text = times_t.text(f"次数：{st.session_state.times}/10")

# 创建一个标题和一个副标题
st.title("💬 InternLM 哄女友模拟器")
st.caption("🚀 EnableAsync 基于 internlm 开发")

# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def get_model():
    # 从预训练的模型中获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True)
    # 从预训练的模型中获取模型，并设置模型参数
    model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    return tokenizer, model

# 加载Chatglm3的model和tokenizer
tokenizer, model = get_model()

# 遍历session_state中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message("user").write(msg[0])
    st.chat_message("assistant").write(msg[1])

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():
    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)

    # 处理逻辑
    scores, mental = get_scores(prompt)
    if scores > 0:
        st.success(f"原谅值+{scores}")
    else:
        st.error(f"原谅值{scores}")
    if scores == -100:
        st.session_state.times += 1
        st.rerun()
    st.session_state.forgiveness += scores
    st.session_state.times += 1

    if st.session_state.forgiveness <= 0:
        response = get_response(prompt, mental)
        st.error("她离开了你，再见！")
        st.button("重新开始")
        st.session_state.forgiveness = 50
        st.session_state.times = 0
        st.session_state["messages"] = [("(哄哄游戏)对方生气了，但你不知道为什么，你需要哄好对方并找到原因", "哼")]
    elif st.session_state.forgiveness >= 100:
        st.success("恭喜恭喜，她原谅了你，你们重归于好了！")
        forgiveness_progress.success("她原谅你了！")
        st.balloons()
        st.button("重新开始")
        st.session_state.forgiveness = 50
        st.session_state.times = 0
        st.session_state["messages"] = [("(哄哄游戏)对方生气了，但你不知道为什么，你需要哄好对方并找到原因", "哼")]
    else:
        response = get_response(prompt, mental)
        # 将模型的输出添加到session_state中的messages列表中
        st.session_state.messages.append((prompt, response))
        # 在聊天界面上显示模型的输出
        st.chat_message("assistant").write(response)

    forgiveness_bar.progress(st.session_state.forgiveness)
    forgiveness_text.text(f"原谅值：{st.session_state.forgiveness}/100")
    times_bar.progress(st.session_state.times * 10)
    times_text.text(f"次数：{st.session_state.times}/10")
    
    if st.session_state.times > 10:
        st.error("她离开了你，再见！")
        st.button("重新开始")
        st.session_state.forgiveness = 40
        st.session_state.times = 0
        st.session_state["messages"] = [("(哄哄游戏)对方生气了，但你不知道为什么，你需要哄好对方并找到原因", "哼")]
    print(st.session_state.messages)
