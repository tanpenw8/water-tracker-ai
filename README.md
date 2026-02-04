Water Intake AI

Water Intake AI is a simple, end-to-end application that helps users track daily water intake, store hydration history, and receive AI-powered hydration feedback.

This project was built to understand how different parts of a real application come together —
from databases and AI agents to APIs and a user interface.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Why this project?

Staying hydrated is important, but most tracking apps feel mechanical and boring.

This project explores:

How AI can give contextual, human-like feedback

How to design a clean backend structure

How data flows from UI → database → AI → UI again

It’s also a hands-on learning project to understand:

Backend fundamentals

API design

AI integration

State handling in frontend apps

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

What does the app do?

At a high level, the app allows a user to:

Log how much water they drank (in ml)

Store that information in a local database

Get hydration advice from an AI assistant

View past intake history in a table and chart

All of this happens in a single, simple dashboard.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


How the project is structured


water_intake.ai/
│
├── src/
│   ├── agent.py        # AI hydration logic
│   ├── database.py     # SQLite database handling
│   ├── logger.py       # Application logging
│   ├── api.py          # Backend logic (API-ready)
│   └── dashboard.py   # Streamlit user interface
│
├── water_tracker.db    # SQLite database (created automatically)
├── app.log             # Log file
├── .env                # Environment variables
└── README.md

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


AI Hydration Agent (agent.py)

This file contains the AI logic of the application.

Uses LangChain with the OpenAI API

Model used: gpt-4o-mini

Takes the user’s water intake as input

Returns friendly hydration feedback

The AI logic is isolated in its own class so it can be reused by:

the Streamlit dashboard

a future FastAPI backend

other services if needed



--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Database Layer (database.py)

The app uses SQLite, a lightweight local database.

Each water intake entry stores:

user_id

intake_ml

intake_date

Main responsibilities:

Create the database table

Insert new intake records

Fetch intake history for a user

The database file is created automatically when the app runs.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Logging is handled using Python’s built-in logging module.

Logs are written to app.log and include:

timestamps

log level (INFO / ERROR)

message details

This helps with:

debugging

tracking user actions

understanding failures without relying on print statements


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


API Layer (api.py)

This file contains backend logic aligned with the database schema.

While it currently mirrors database operations, it is intentionally structured so it can be:

easily converted into a FastAPI REST API

used by multiple frontends in the future

This keeps business logic separate from the UI.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Streamlit Dashboard (dashboard.py)

The dashboard is built using Streamlit and acts as the frontend.

What the dashboard includes:

Welcome screen with session tracking

Sidebar to log water intake

Instant AI feedback after logging

Water intake history table

Line chart showing intake trends over time

Streamlit reruns the script on every interaction, so session_state is used to maintain app flow.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Tech Stack

Python 3.12

Streamlit – frontend UI

SQLite – database

LangChain – AI orchestration

OpenAI API – hydration feedback

Pandas – data manipulation

Logging – application monitoring
