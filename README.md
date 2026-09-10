# Hiver AI Customer Support Agent

An AI-assisted customer support agent built as a take-home SDE assignment.

The system analyzes historical customer-support conversations, classifies incoming customer issues, retrieves similar historical support cases, and decides whether to answer, clarify, or escalate.

---

## 1. Problem

Customer-support teams handle large volumes of repetitive requests across areas such as:

- Account access
- Billing and payments
- Premium subscriptions
- Playback problems
- Application issues
- Device/platform issues
- Content availability
- Playlists and libraries
- Feature requests
- Account security

The goal of this project is to build a small, explainable support-agent pipeline that uses historical support interactions while avoiding unsupported answers when the available evidence is weak.

---

## 2. Approach

The overall pipeline is:

```text
Historical Support Dataset
          |
          v
Data Audit & Cleaning
          |
          v
Conversation Reconstruction
          |
          v
Brand Selection
          |
          v
SpotifyCares Support Data
          |
          v
Intent Discovery
          |
          v
Human-Labeled Golden Set
          |
          +--------------------+
          |                    |
          v                    v
   Intent Classifier     Historical Retriever
          |                    |
          +---------+----------+
                    |
                    v
             Decision Engine
                    |
          +---------+---------+
          |         |         |
          v         v         v
       ANSWER   CLARIFY   ESCALATE
                    |
                    v
              FastAPI API