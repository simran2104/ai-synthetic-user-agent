# 🛋️ Furniture Mart - Sarvotam Furniture
### A home with a story.

**Sarvotam Furniture** is a Django storefront where users can browse furniture products, view product details, manage a showroom shortlist, create accounts, and submit enquiries.

🌐 **Live Website:** [furniture-mart-ps3p.onrender.com](https://furniture-mart-ps3p.onrender.com/)

GA4 page-view tracking is configured in the shared `templates/base.html` template, so it loads on pages that inherit the base. The Measurement ID is defined there once and reused by the Google tag configuration; no custom events are configured.

---

## ✨ Features

* 🛋️ Browse furniture products
* 🔎 View detailed product information
* 🛒 Add products to the shopping cart
* ➕ Update product quantities in the cart
* 👤 User registration and authentication
* 🏬 Request showroom visits and product enquiries
* 📱 Responsive and user-friendly interface
* 🔐 Django-based backend and authentication
* 🗄️ Database-driven product and order management

---

## 🖥️ Tech Stack

### Backend

* **Python**
* **Django**

### Frontend

* **HTML5**
* **CSS3**
* **JavaScript**
* **Bootstrap**

### Database

* **SQLite** for development

### Deployment

* **Render**

### Version Control

* **Git**
* **GitHub**

---

## 🚀 Getting Started

Follow these steps to run Sarvotam Furniture locally.

### 1. Clone the repository

```bash
git clone https://github.com/simran2104/furniture-mart.git
cd furniture-mart
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Apply database migrations

```bash
python manage.py migrate
```

### 5. Create a superuser

```bash
python manage.py createsuperuser
```

### 6. Start the development server

```bash
python manage.py runserver
```

Open your browser and visit:

```text
http://127.0.0.1:8000/
```

## Synthetic Browser Agent

The standalone `ai_agent/` package uses Gemini with Playwright to observe and interact with the Furniture Mart site. It does not complete purchases. The storefront supports showroom enquiries, but has no online checkout, payment, or order workflow.

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

Set `GEMINI_API_KEY` in your shell environment before using Gemini. Never put the key in source files, JSON settings, documentation, or version control. Gemini usage is subject to the quota for your Google AI Studio project.

Run the structured Gemini smoke test without launching a browser:

```bash
python main.py --gemini-test
```

### Browser Observation Smoke Test

Install the Playwright Chromium browser once, if not already installed:

```bash
python -m playwright install chromium
```

Run the browser smoke test from the repository root:

```bash
python main.py --browser-test
```

The smoke test launches Chromium, opens the configured Furniture Mart homepage, waits for the Render service if it is waking, and prints the URL, title, and a compact observation of headings, links, buttons, inputs, visible text, and product cards. It does not call Gemini or interact with the site. Screenshots are saved under `agent_data/sessions/` only when `screenshots` is `true` in `agent_config/settings.json`.

Example output (product data and screenshot filename vary):

```text
URL: https://furniture-mart-ps3p.onrender.com/
Title: Sarvotam Furniture | A home with a story
Observation:
{
	"headings": ["Furniture for every corner of your home.", "Find your room"],
	"links": [{"label": "Shop", "url": "https://furniture-mart-ps3p.onrender.com/products/"}],
	"buttons": [{"label": "Accept", "type": "button"}],
	"inputs": [{"name": "q", "type": "text", "placeholder": "Search furniture..."}],
	"products": [{
		"name": "Studio Transitional Dining Set",
		"price": "₹2355",
		"original_price": "₹3855",
		"url": "https://furniture-mart-ps3p.onrender.com/product/studio-transitional-dining-set-16/"
	}]
}
Screenshot: agent_data/sessions/browser-<timestamp>.png
```

Browser behavior is controlled in `agent_config/settings.json`. `headless: false` (the development default) shows Chromium; set it to `true` for a background run. `startup_timeout_seconds` controls the total wait for the site, and `startup_retry_interval_seconds` controls the delay between attempts. The configured browser destination is restricted to `https://furniture-mart-ps3p.onrender.com/`.

Render's free service can sleep while idle. During a cold start, it may return HTTP 503 or a page titled “Application loading” while the instance wakes. The smoke test recognizes those responses and retries until the configured timeout; if the site remains unavailable, it reports the last status and title instead of printing a successful observation.

Run the browser unit tests without contacting the live site:

```bash
python -m unittest discover -s tests -v
```

Troubleshooting:

* If Playwright reports that its Chromium executable is missing, rerun `python -m playwright install chromium`.
* If the smoke test reaches its startup timeout with HTTP 503, try again later or increase `startup_timeout_seconds`; the Render instance may still be waking or unavailable.
* If Chromium cannot open in headed mode, check that a desktop session is available or set `headless` to `true`.

### Gemini Browser Agent

Gemini is the only supported LLM provider. The model is configured in `agent_config/settings.json` as `gemini-3.8-flash`.

Set `GEMINI_API_KEY` in the process environment before running Gemini commands. Do not put the key in source files, JSON settings, documentation, or version control. The key is not stored by the application. Gemini API access and quota depend on your Google AI Studio project and applicable usage limits.

Run the structured Gemini smoke test (no browser is launched):

```bash
python main.py --gemini-test
```

After the smoke test returns a valid structured action, run one headed browser session with the read-only persona:

```bash
python main.py --agent-test --persona casual_browser
```

Gemini receives exactly one function, `next_action`. The Python agent validates arguments and action permissions before dispatching to Playwright. Chromium is visible by default (`headless: false`). Cart actions remain disabled unless both global configuration and persona permission allow them. No checkout, payments, or online orders are supported.

### Diagnose Structured Tool Calls

To probe the active provider against a real homepage observation, run:

```bash
python diagnose_llm.py
```

The diagnostic supplies one `next_action` schema and a fresh live homepage observation. It reports PASS only for exactly one schema-valid structured call; an HTTP success by itself is not sufficient. Gemini logs record request metadata only and do not capture credentials or raw authenticated requests.

