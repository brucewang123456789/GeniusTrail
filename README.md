# Veltraxor

Veltraxor is a persona-driven chatbot built for **dynamic chain-of-thought (D-CoT)** reasoning.  
Inspired by the D-CoT framework proposed in *Dynamic Chain-of-Thought: Towards Adaptive Deep Reasoning*, the bot decides on-the-fly how many reasoning steps to take, balancing speed on trivial queries with depth on complex problems.

---

## Features

| Capability | Description |
|------------|-------------|
| **Adaptive Reasoning** | After every provisional answer, a lightweight critic judges whether another reasoning pass will improve the result. Easy questions halt early; hard ones trigger extended reflection. |
| **Modular Architecture** | Designed for future integration of tone control, memory retrieval, and CoT evaluation components. |
| **Token-Efficient** | D-CoT automatically prunes unnecessary thought steps, reducing latency and API spend without sacrificing accuracy. |
| **API-First** | All model calls are routed through a cloud language-model endpoint; no local GPU or heavyweight runtime needed. |

---

## Quick Start

```bash
git clone https://github.com/your-handle/veltraxor.git
cd veltraxor
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py    # returns a short reply if connectivity is healthy
