## 🤖 AI Web Analytics Agent

An AI-powered synthetic user agent that simulates realistic website behaviour using Google Gemini, Python, and Playwright.

The agent uses different user personas to browse a Django furniture website and generates session data that can be used for web analytics.

🌐 **Target Website:** https://furniture-mart-ps3p.onrender.com/

---

### 🏗️ System Architecture

<img width="2889" height="654" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/3db8cc37-8bb5-44a7-8251-559d5d614847" />

#### How it works
1. A persona defines the user's goal.
2. Python opens the website using Playwright.
3. The agent observes the current page.
4. Gemini generates a structured session_plan.
5. Python validates the complete plan.
6. Playwright executes the actions.
7. Session activity is recorded for analytics.

---

### 🧠 AI Agent

Gemini generates structured browser actions, while Python remains the control and safety layer.

Python acts as the control and safety layer between the LLM and the browser.

#### Python validates:
- Allowed actions
- URLs and website origin
- Observed links and products
- Persona permissions
- Cart permissions
- Session limits
- Repeated actions

---

### 👥 Synthetic User Personas

The agent can run sessions using different synthetic user behaviours.

```text
Casual Browser
Product Researcher
Goal-Oriented Buyer
Potential Buyer
```

Each persona can have a different objective and browsing style.

This allows the same website to be explored from multiple simulated user perspectives.

---

### 📊 Web Analytics

The project combines two sources of analytics.

#### Synthetic Agent Analytics

The agent records information from its synthetic sessions, including:

- User persona
- Actions performed
- Pages visited
- Product views
- Searches
- Cart interactions
- Session duration
- Number of actions
- Journey path
- Session outcome

This data can be used to analyse how different synthetic users navigate the website.

#### Google Analytics 4

Google Analytics 4 is configured on the target website for page-view analytics.

GA4 provides an independent view of website activity, while the synthetic-agent logs provide detailed information about the agent's decisions and journeys.

---

### 🌐 Target Web Application

The current experiment uses **Sarvotam Furniture (Furniture Mart)** as the target web application.

The website is a Django-based furniture storefront that allows visitors to:

- Browse furniture products
- Search for products
- View product details
- Add products to a shopping cart
- Manage cart quantities
- Create accounts
- Submit showroom and product enquiries

The website serves as the **environment in which the synthetic users operate**.

The primary focus of this repository is the AI agent and the web analytics experiment, rather than the storefront itself.

🌐 **Live Website:** https://furniture-mart-ps3p.onrender.com/

---

### 🚀 Getting Started

#### 1. Clone the repository

```bash
git clone https://github.com/simran2104/ai-web-analytics-agent.git
cd ai-web-analytics-agent
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Install Playwright Chromium

```bash
python -m playwright install chromium
```

#### 4. Configure Gemini

Set the Gemini API key in your shell environment.

##### Windows PowerShell

```powershell
$env:GEMINI_API_KEY="YOUR_KEY_HERE"
```

The Gemini API key is read from the `GEMINI_API_KEY` environment variable.

---

### 🧪 Testing

#### Gemini Smoke Test

Run the structured Gemini test without launching a browser:

```bash
python main.py --gemini-test
```

This verifies that Gemini can return a valid structured `session_plan`.

---

#### 🌐 Browser Smoke Test

Run the Playwright browser test:

```bash
python main.py --browser-test
```

This test:

- Launches Chromium
- Opens the configured Furniture Mart website
- Handles Render cold starts
- Waits for the website to become available
- Extracts a compact page observation
---

### 🤖 Run the AI Agent

After the Gemini smoke test succeeds, run one synthetic-user session:

```bash
python main.py --agent-test --persona casual_browser
```

Chromium is visible by default.

To run the browser in the background, set:

```json
"headless": true
```

in:

```text
agent_config/settings.json
```

Gemini receives exactly one structured function:

```text
session_plan
```

The Python agent validates the returned action and then dispatches it to Playwright.

Cart actions remain disabled unless both global configuration and persona permissions allow them.

The agent does not perform checkout, payments, or online purchases.

---

### 🛠️ Technology Stack

#### AI & Agent

- Python
- Google Gemini - gemini-3.5-flash-lite
- Structured Function Calling

#### Browser Automation

- Playwright
- Chromium

#### Target Web Application

- Django
- Python
- HTML5
- CSS3
- JavaScript
- Bootstrap
- SQLite

#### Analytics

- Google Analytics 4
- Synthetic session/event logs

#### Deployment

- Render

#### Version Control

- Git
- GitHub
