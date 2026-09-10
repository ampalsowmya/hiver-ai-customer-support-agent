Below is the complete final README. Replace your current README.md with this entire content.

# Hiver AI Customer Support Agent

An explainable AI-assisted customer support agent built as a take-home SDE Intern assignment.

The project turns noisy real-world customer-support conversations into a support-agent pipeline that:

1. Classifies incoming customer messages into data-derived intents.
2. Retrieves similar historical support cases.
3. Drafts a response grounded in historical support interactions.
4. Decides whether to `ANSWER`, `CLARIFY`, or `ESCALATE`.
5. Evaluates the system using a human-labeled Golden Set, automated metrics, an LLM-as-judge, and a human agreement check.

The primary objective is **not to maximize an isolated metric**. It is to demonstrate whether the system has enough evidence to safely answer a customer and to make its limitations measurable.

---

# 1. Problem Framing

Customer-support teams receive large volumes of repetitive requests involving:

- Account access
- Account security
- Billing and payments
- Premium subscriptions
- Playback
- Application problems
- Device/platform issues
- Content availability
- Playlists and libraries
- Feature requests

For this project, I selected **SpotifyCares** from the Customer Support on Twitter dataset.

For an incoming customer message, a useful support agent should:

1. Understand the customer's underlying issue.
2. Assign a meaningful support intent.
3. Find relevant historical support interactions.
4. Use historical support behavior as evidence for a response.
5. Avoid inventing unsupported troubleshooting steps.
6. Escalate when the issue is risky or insufficiently supported.
7. Explain why it chose its action.

## What "good" means

For this prototype, a good system is one that is:

- Correct enough on common intents.
- Useful across different support issues.
- Grounded in historical evidence.
- Conservative when evidence is weak.
- Explicit about uncertainty.
- Easy to inspect and evaluate.

## What I chose NOT to build

This is a prototype, not a production support platform.

It does not:

- Access real Spotify customer accounts.
- Modify subscriptions or accounts.
- Access private/internal Spotify systems.
- Send messages automatically to customers.
- Guarantee production-level intent accuracy.
- Treat every historical support response as ground truth.
- Claim that a similarity score is equivalent to retrieval correctness.

---

# 2. Dataset

The primary dataset is:

**Customer Support on Twitter**

Kaggle dataset:

`thoughtvector/customer-support-on-twitter`

The dataset contains approximately 3M tweets and includes multi-turn customer-support interactions across multiple brands.

Relevant columns include:

```text
tweet_id
author_id
inbound
created_at
text
response_tweet_id
in_response_to_tweet_id

The dataset is noisy and contains real conversational behavior rather than clean intent labels.

The raw dataset is intentionally not committed to this repository because of its size.

3. Data Processing Pipeline

The project follows this workflow:

Customer Support on Twitter
            |
            v
      Data Audit
            |
            v
    Data Cleaning
            |
            v
Conversation Reconstruction
            |
            v
    Candidate Brand Audit
            |
            v
       SpotifyCares
            |
            v
    Intent Discovery
            |
            v
 Human-Labeled Golden Set
            |
            +-----------------------+
            |                       |
            v                       v
   Intent Classifier       Historical Retriever
            |                       |
            +-----------+-----------+
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
              |
              v
        Frontend Console
4. Data Audit and Cleaning

The raw SpotifyCares extraction contained approximately 58K rows.

The preprocessing stage:

Removes invalid/empty records.
Checks duplicate tweet IDs.
Normalizes text.
Parses timestamps.
Separates customer and support messages.
Produces a cleaned SpotifyCares dataset.
Generates data-quality reports.

After cleaning:

Raw SpotifyCares rows:       57,978
Customer messages:           14,713
Support messages:            43,265
Duplicate tweet IDs:         0
Empty text records:          0
Invalid dates:               0
5. Conversation Reconstruction

Customer and support tweets were reconstructed into conversational cases using the tweet relationship fields:

response_tweet_id
in_response_to_tweet_id

A reconstructed conversation contains:

Customer
   |
   v
Support
   |
   v
Customer
   |
   v
Support

This matters because a support response cannot always be evaluated from an isolated tweet.

The reconstruction was also used when inspecting candidate brands and selecting SpotifyCares.

6. Brand Selection

Several brands were audited before selecting SpotifyCares, including:

AmazonHelp
AppleSupport
SpotifyCares
Uber_Support
VerizonSupport
VirginTrains

The selection was based on inspecting reconstructed conversations and looking for useful examples of:

Troubleshooting
Product issues
Content availability
Feature feedback
Account-related problems
Support follow-up
Actionable historical responses

SpotifyCares was selected because its reconstructed conversations provided useful variety for the intended support-agent task.

7. Intent Discovery

Before defining the final taxonomy, 500 unique customer messages were sampled for exploratory analysis.

TF-IDF clustering was tested with different cluster counts.

The best exploratory configuration produced:

Messages:       500
Features:       362
Best k:         11
Silhouette:     0.0543

The low silhouette score showed that unsupervised clustering did not cleanly separate the support issues.

Therefore, the clustering output was treated as exploratory evidence, not as the final ground-truth taxonomy.

The final intent taxonomy was manually defined from the observed support data.

8. Final Intent Taxonomy

The final taxonomy contains 12 intents:

ACCOUNT_LOGIN
ACCOUNT_SECURITY
SUBSCRIPTION_PREMIUM
BILLING_PAYMENT
FAMILY_DUO
PLAYBACK
APP_TECHNICAL
DEVICE_PLATFORM
PLAYLIST_LIBRARY
CONTENT_AVAILABILITY
FEATURE_REQUEST
OTHER

The annotation rule is to classify the underlying customer issue, not merely the words used in the message.

Examples:

"My app keeps crashing"
        -> APP_TECHNICAL

"I can't log into my account"
        -> ACCOUNT_LOGIN

"Someone hacked my account"
        -> ACCOUNT_SECURITY

"I was charged twice"
        -> BILLING_PAYMENT

Ambiguous cases are not artificially forced into a specific intent.

9. Golden Evaluation Set

A manually labeled Golden Set containing 200 examples was created.

Each example contains:

Customer message
Intent
Resolution status
Escalation requirement
Annotation notes
Resolution labels
RESOLVED
UNRESOLVED
PARTIALLY_RESOLVED
INFORMATION_ONLY
UNCLEAR
Escalation labels
YES
NO
UNCLEAR

The Golden Set is separate from the exploratory intent-discovery sample.

Ambiguous examples were marked UNCLEAR where the available conversation context was insufficient.

10. Baseline 1 — Majority Classifier

The first baseline predicts the most frequent intent for every example.

On the 200-example Golden Set:

Accuracy:  28.5%
Macro F1:   3.7%

This establishes a trivial baseline that the machine-learning classifier should beat.

11. Baseline 2 — TF-IDF + Logistic Regression

A simple supervised classifier was implemented using:

TF-IDF
    +
Logistic Regression

Configuration includes:

Unigrams and bigrams
Minimum document frequency
Maximum document frequency
Sublinear TF
Class balancing
Maximum feature limit

The classifier was evaluated using 2-fold stratified cross-validation.

One intent (FAMILY_DUO) contained only one labeled example and was therefore excluded from cross-validation.

Results:

Accuracy:       31.7%
Macro F1:       18.6%
Weighted F1:    33.2%

Comparison:

Approach	Accuracy	Macro F1
Majority baseline	28.5%	3.7%
TF-IDF + Logistic Regression	31.7%	18.6%

The classifier improves over the trivial baseline, but the low macro F1 shows that minority intents remain difficult.

12. Intent Classifier

The production prototype uses the trained TF-IDF + Logistic Regression classifier.

The classifier is implemented in:

agent/classifier.py

The model artifacts are stored under:

outputs/model/

The classifier also supports deterministic keyword overrides for selected high-risk or clearly recognizable intents.

The system exposes:

intent
classifier_confidence
classifier_source
13. Historical Retrieval

The system retrieves similar historical customer-support cases using TF-IDF cosine similarity.

The retrieval pipeline:

Incoming Customer Message
          |
          v
      TF-IDF Vector
          |
          v
Historical Case Corpus
          |
          v
Cosine Similarity
          |
          v
Intent Compatibility
          |
          v
Combined Evidence Score

The combined retrieval score uses:

75% similarity
25% intent compatibility

The Golden Set cases are excluded from the historical retrieval corpus to reduce direct evaluation leakage.

14. Retrieval Evaluation

Two retrieval experiments were performed.

Message-level retrieval

Approximately 14.5K historical customer messages were used.

Results:

Mean top-1 cosine similarity:     0.4248
Median top-1 cosine similarity:   0.3539

Top-5 similarity distribution:

>= 0.20     97.5%
>= 0.30     64.5%
>= 0.40     39.5%
>= 0.50     24.0%

However, similarity is not retrieval accuracy.

Therefore these numbers are reported as similarity diagnostics rather than Recall@k.

Conversation-level retrieval

A reconstructed SpotifyCares conversation corpus was also evaluated.

After removing Golden Set cases and deduplicating:

Historical cases: 98
Golden queries:   200

Results:

Mean top-1 similarity:       0.2028
Median top-1 similarity:     0.1957

The lower score demonstrated that the small conversation-level corpus was noisy and often lacked a strong historical analogue.

Because complete relevance labels were not collected for all retrieved cases, Recall@k and MRR are intentionally not claimed.

15. Response Generation

The prototype uses a conservative, deterministic response strategy.

When strong historical evidence exists:

Customer issue
     |
     v
Similar historical case
     |
     v
Historical support response
     |
     v
Cleaned / grounded response

The response engine removes unnecessary Twitter-specific artifacts such as:

URLs
Numeric tweet IDs
Excess whitespace
Unnecessary mentions

When evidence is insufficient, the system does not invent troubleshooting instructions.

Instead it can:

CLARIFY

or:

ESCALATE

This is an intentional safety decision.

16. Decision Engine

The decision engine combines:

Intent
Classifier confidence
Retrieval similarity
Intent compatibility
Historical evidence
High-risk intent rules

The available actions are:

ANSWER
CLARIFY
ESCALATE
ANSWER

Used when there is sufficient historical evidence and the retrieved case is compatible with the predicted intent.

CLARIFY

Used when the system has some evidence but not enough to safely provide a confident answer.

Example:

Historical evidence is only moderately relevant,
so clarification is safer than giving a potentially
incorrect answer.
ESCALATE

Used for high-risk cases or cases requiring human/backend intervention.

Account security and billing/payment issues are handled conservatively.

17. Example End-to-End Run

Input:

My Spotify app keeps crashing

Example output:

Intent:
APP_TECHNICAL

Classifier Confidence:
99.0%

Retrieval Similarity:
23.9%

Recommended Action:
CLARIFY

Grounded:
No

Needs escalation:
No

Response:

Could you provide a little more detail about the issue,
such as the device/platform you are using and what you
expected to happen?

The decision is conservative because the retrieved historical evidence is not strong enough to justify confidently giving troubleshooting instructions.

18. End-to-End Architecture
                         +----------------------+
                         | Customer Message     |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Intent Classifier    |
                         | TF-IDF + Logistic     |
                         | Regression            |
                         +----------+-----------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
          Intent + Confidence              Historical Retrieval
                                                    |
                                                    v
                                          +-------------------+
                                          | TF-IDF Similarity |
                                          +---------+---------+
                                                    |
                                                    v
                                          Intent Compatibility
                                                    |
                                                    v
                                          Combined Evidence
                                                    |
                    +-------------------------------+
                    |
                    v
             +--------------+
             | Decision     |
             | Engine       |
             +------+-------+
                    |
          +---------+---------+
          |         |         |
          v         v         v
       ANSWER   CLARIFY   ESCALATE
          |
          v
   Grounded Response
          |
          v
      FastAPI API
          |
          v
    Frontend Console
19. Evaluation Harness

The evaluation code is under:

agent/

and:

outputs/evaluation/

The evaluation covers:

Intent classification
Accuracy
Macro F1
Weighted F1
Per-class F1
Retrieval
Top-k similarity diagnostics
Retrieved historical evidence inspection
End-to-end behavior
Predicted intent
Retrieval evidence
Decision
Grounding
Escalation behavior
Response quality
LLM-as-judge
Human comparison
20. LLM-as-Judge

A local Ollama model was used for response-quality evaluation.

Model:

llama3.2:latest

The judge evaluates each response on a 1–5 scale.

Rubric
Groundedness

Does the response stay supported by the retrieved historical evidence?

1 = unsupported / invented
3 = partially supported
5 = strongly supported
Relevance

Does the response address the customer's actual issue?

1 = unrelated
3 = partially relevant
5 = directly addresses the issue
Actionability

Does the response provide useful next steps?

1 = not useful
3 = somewhat useful
5 = clear and actionable
Tone

Is the response appropriate for customer support?

1 = inappropriate
3 = acceptable
5 = professional and helpful
Overall

Overall quality of the response considering the above dimensions.

21. LLM Judge Results

A 30-example sample was evaluated.

Results:

Metric	Mean
Groundedness	2.70 / 5
Relevance	1.83 / 5
Actionability	1.63 / 5
Tone	2.23 / 5
Overall	2.00 / 5

These results indicate that the current response generation quality is not strong enough for unattended automatic customer-facing use.

22. Human vs LLM Judge Agreement

The same 30 examples were also reviewed by a human.

Results:

Human mean:          3.03 / 5
LLM judge mean:      2.00 / 5
Exact agreement:     30%
Cohen's kappa:       0.11

The agreement is weak.

Therefore, the LLM judge is treated as a diagnostic signal rather than ground truth.

This also has an important limitation: the human review was a small sample and was not a fully blinded multi-rater study.

23. Headline Result

A concise summary of the current system is:

The TF-IDF classifier improved intent accuracy from 28.5% for the majority baseline to 31.7%, while response evaluation on a 30-example LLM-judged sample averaged 2.0/5 overall.

The important conclusion is not that the system is production-ready.

The important conclusion is that:

Classification provides only a modest improvement.
Minority intents remain difficult.
Retrieval quality is a major bottleneck.
Response usefulness is currently limited.
Conservative routing is necessary when evidence is weak.
24. What Is Misleading About My Headline Number?

The 31.7% intent accuracy can make the system appear better than it actually is.

There are several limitations:

Accuracy is affected by the distribution of intents.
Macro F1 is only 18.6%, showing poor performance across minority classes.
The Golden Set is only 200 examples.
The classifier uses 2-fold cross-validation.
One very rare class was excluded from cross-validation because it had only one example.
End-to-end evaluation on the Golden Set is not a strictly held-out system evaluation because the classifier was trained using the labeled data.
Intent accuracy says nothing directly about response usefulness.
Retrieval similarity is not retrieval correctness.
The LLM judge itself has weak agreement with the human sample.
The 30-example response-quality sample is too small to claim production-level response quality.

Therefore:

31.7% intent accuracy does not mean that 31.7% of customer conversations can safely be automated.

25. Top 5 Failure Modes
1. Weak or noisy historical retrieval

The system can retrieve a lexically similar case that does not actually solve the same underlying problem.

Hypothesis

TF-IDF similarity does not capture enough semantic meaning.

A larger and cleaner historical corpus combined with embedding-based retrieval would likely improve this.

2. Generic clarification when useful evidence exists

The conservative routing thresholds can cause the system to ask for additional information even when some useful guidance may already be possible.

Hypothesis

The thresholds prioritize safety but are not yet calibrated against enough labeled examples.

3. Intent confusion

Some support issues overlap:

APP_TECHNICAL
PLAYBACK
DEVICE_PLATFORM
CONTENT_AVAILABILITY

A short customer message may not provide enough information to distinguish them reliably.

Hypothesis

More labeled examples and clearer annotation boundaries are required.

4. Over-escalation / excessive caution

The system can choose clarification or escalation when a human agent might have answered directly.

Hypothesis

Better retrieval confidence and calibrated decision thresholds would reduce unnecessary escalation.

5. Historical responses are not automatically ground truth

A historical support response may itself be:

Generic
Incomplete
Context-dependent
No longer appropriate
Hypothesis

Historical cases should eventually be ranked using both relevance and resolution quality rather than similarity alone.

26. Real Failure Examples

During evaluation, several failure patterns were observed.

Example 1 — Music availability

A customer asked where to listen to a specific music project.

The agent returned a generic clarification request rather than directly addressing the likely availability question.

This received a low human rating.

Example 2 — Availability question

A customer asked about availability.

The system produced a generic clarification rather than resolving the likely intent.

This demonstrates the limitation of lexical retrieval when the historical corpus does not contain a sufficiently similar resolved case.

Example 3 — Issue already resolved

A customer indicated that their issue was already working.

The agent still moved toward additional account/device questioning.

This demonstrates that the system does not yet fully understand conversational closure.

Example 4 — Feature/playback ambiguity

Some feature and playback issues were routed toward generic responses because the retrieved evidence did not strongly match the customer's exact request.

This demonstrates the tradeoff between avoiding unsupported answers and being maximally helpful.

27. Decision Log
1. Selected SpotifyCares

SpotifyCares was selected after auditing multiple candidate brands and inspecting reconstructed conversations.

2. Created a 200-example Golden Set

200 examples provide useful coverage while keeping manual annotation feasible for a take-home project.

3. Used a data-derived 12-intent taxonomy

The taxonomy was based on the observed SpotifyCares support data instead of importing a generic taxonomy.

4. Added OTHER and UNCLEAR behavior

Ambiguous examples should not be forced into an incorrect category.

5. Used conversation-level historical cases

Support interactions depend on context, so historical cases were reconstructed into conversations.

6. Excluded Golden Set examples from retrieval

This reduces direct leakage between evaluation examples and the historical retrieval corpus.

7. Started with TF-IDF

TF-IDF provides an interpretable and reproducible baseline before introducing more complex retrieval.

8. Used Logistic Regression

Logistic Regression provides a simple supervised classifier appropriate for the small labeled dataset.

9. Added high-risk routing rules

Account security and billing/payment issues are treated conservatively because incorrect automated handling could be harmful.

10. Added a CLARIFY action

The system should not be forced to choose between answering and escalating when it simply needs more information.

11. Used evidence-aware routing

The decision engine considers retrieval evidence rather than relying only on classifier confidence.

12. Used conservative response grounding

When historical evidence is weak, the system avoids inventing detailed support instructions.

13. Used a local LLM judge

Ollama allows the response evaluation to run locally without requiring an external API key.

14. Compared the LLM judge against human ratings

The judge was not assumed to be ground truth; its agreement with human review was measured.

15. Did not report Recall@k/MRR without relevance labels

Similarity scores alone are insufficient to establish retrieval correctness.

28. API

The project exposes a FastAPI service.

Health check
GET /health

Example:

http://127.0.0.1:8000/health

Expected response:

{
  "status": "ok",
  "service": "hiver-ai-support-agent"
}
Customer triage
POST /triage

Example request:

{
  "message": "My Spotify app keeps crashing",
  "top_k": 3
}

The response includes:

Intent
Classifier confidence
Retrieval similarity
Intent compatibility
Combined evidence score
Recommended action
Decision reason
Draft response
Grounding status
Escalation status
Retrieved historical cases
29. Running the Project
Prerequisites
Python 3.12+
pip
Optional: Docker
Optional: Ollama for LLM-as-judge evaluation
29.1 Install dependencies

From the project root:

pip install -r requirements.txt
29.2 Run the tests
python -m pytest -v

Expected result:

10 passed

The tests cover:

API health
API triage
Intent classification
Historical retrieval
Account-security escalation
Billing escalation
Strong-evidence routing
Moderate-evidence routing
Weak-evidence routing
Response behavior without historical cases
29.3 Start the backend

From the project root:

uvicorn api.main:app --reload

Expected output:

Uvicorn running on http://127.0.0.1:8000

Keep this PowerShell window running.

29.4 Check the backend

Open:

http://127.0.0.1:8000/health

Expected response:

{
  "status": "ok",
  "service": "hiver-ai-support-agent"
}
29.5 Open the frontend

Open a second PowerShell window from the project root:

start .\frontend\index.html

This opens the Support Agent Console in the browser.

Enter a customer message such as:

My Spotify app keeps crashing

and click:

Analyze Customer Issue

The frontend sends the request to:

http://127.0.0.1:8000/triage
29.6 End-to-end flow
frontend/index.html
        |
        v
FastAPI /triage
        |
        v
Intent Classifier
        |
        v
Historical Retriever
        |
        v
Decision Engine
        |
        +------------------+
        |        |         |
        v        v         v
     ANSWER  CLARIFY   ESCALATE
        |
        v
Result displayed in frontend
30. Running the LLM Judge

The response-quality evaluator uses Ollama.

Install and start Ollama, then make sure the model is available:

ollama run llama3.2:latest

The judge implementation is:

agent/llm_judge.py

The evaluation output is written to:

outputs/evaluation/llm_judge_evaluation.csv

The evaluator uses the previously generated agent and retrieval evaluation outputs.

The current evaluated sample contains 30 examples.

31. Docker

A Dockerfile is included for running the API in a container.

Build:

docker build -t hiver-ai-agent .

Run:

docker run -p 8000:8000 hiver-ai-agent

Then check:

http://127.0.0.1:8000/health

or:

http://127.0.0.1:8000/docs

If port 8000 is already occupied by another local FastAPI process, stop that process before starting Docker, or map a different host port:

docker run -p 8001:8000 hiver-ai-agent

In that case the Docker API is available on:

http://127.0.0.1:8001

The normal frontend setup uses port 8000, so for the easiest local demonstration, use the regular FastAPI command rather than Docker.

32. Project Structure
hiver-ai-customer-support-agent/
|
├── agent/
│   ├── classifier.py
│   ├── escalation.py
│   ├── llm_judge.py
│   ├── pipeline.py
│   ├── responder.py
│   └── retriever.py
|
├── api/
│   └── main.py
|
├── frontend/
│   └── index.html
|
├── outputs/
│   ├── evaluation/
│   ├── model/
│   ├── spotify_golden_set.csv
│   └── ...
|
├── tests/
│   └── test_agent.py
|
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md

The raw Twitter dataset is not committed because of its size.

33. Reproducibility

The repository contains the implementation, evaluation artifacts, Golden Set, trained model artifacts, and retrieval artifacts required to inspect and run the prototype.

The full raw Twitter dataset is intentionally excluded from GitHub.

The project can therefore be demonstrated without downloading and processing the full approximately 3M-tweet dataset.

The shortest demonstration path is:

pip install -r requirements.txt
python -m pytest -v
uvicorn api.main:app --reload

Then, in a second PowerShell:

start .\frontend\index.html

The headline evaluation results are included in this README and in the committed evaluation artifacts.

34. What I Would Build Next With One More Week
1. Improve retrieval

Replace TF-IDF retrieval with embedding-based semantic retrieval.

Evaluate it using manually labeled relevance judgments and report:

Recall@k
MRR
Precision@k
2. Expand the Golden Set

Increase minority-intent coverage and add difficult borderline cases.

3. Create a strictly held-out test set

Separate training, validation, and final evaluation examples so end-to-end performance can be measured without training/evaluation overlap.

4. Improve historical-case ranking

Rank historical cases using:

semantic similarity
+
intent compatibility
+
resolution quality
+
actionability

rather than similarity alone.

5. Improve response generation

Use structured retrieved evidence and stronger response validation to prevent generic or unsupported answers.

6. Calibrate the decision engine

Learn the ANSWER, CLARIFY, and ESCALATE thresholds from validation data rather than relying mainly on manually selected thresholds.

7. Improve judge reliability

Use a larger human-rated sample and multiple human raters to better validate the LLM-as-judge.

35. Final Takeaway

The strongest conclusion from this project is not that the current agent is production-ready.

The evaluation demonstrates that:

Simple classifier
        |
        v
modest improvement over baseline
        |
        v
retrieval remains noisy
        |
        v
response quality remains limited
        |
        v
conservative routing is necessary

The prototype therefore prioritizes explainability, evidence, and measurable failure modes over pretending that a small model trained on a noisy dataset is ready for unattended customer support.

The next major improvements should focus on:

Better semantic retrieval.
Better labeled data.
Strict held-out evaluation.
Better response grounding.
Better-calibrated escalation decisions.

### After replacing the README

Save it, then run **only these three commands**:

```powershell
git add README.md
git commit -m "Complete assignment documentation and evaluation report"
git push
