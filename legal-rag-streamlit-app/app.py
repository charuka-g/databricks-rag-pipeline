"""
Legal RAG Assistant - Databricks App (Streamlit)

Three tabs: RAG Chat, Feedback, Usage & Cost.

Calls the legal-rag-endpoint serving endpoint, records token usage in
workspace.legal.rag_usage, and records user feedback in workspace.legal.rag_feedback.

Endpoint name and warehouse ID come from app resources as environment variables.
No tokens or credentials in this file.
"""

import json
import os
import time
import uuid

import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementParameterListItem

# ---------------------------------------------------------------------------
# Config - from app resources / environment
# ---------------------------------------------------------------------------
ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME", "legal-rag-endpoint")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")

CATALOG = os.getenv("RAG_CATALOG", "workspace")
SCHEMA = os.getenv("RAG_SCHEMA", "legal")
USAGE_TABLE = f"{CATALOG}.{SCHEMA}.rag_usage"
FEEDBACK_TABLE = f"{CATALOG}.{SCHEMA}.rag_feedback"

# Estimated pricing assumptions, not Databricks pricing. Default to 0.0 so the app never
# shows an invented cost. Set them in app.yaml when you have your real rates.
INPUT_COST_PER_1M_TOKENS = float(os.getenv("INPUT_COST_PER_1M_TOKENS", "0.0"))
OUTPUT_COST_PER_1M_TOKENS = float(os.getenv("OUTPUT_COST_PER_1M_TOKENS", "0.0"))

# The SDK reads DATABRICKS_HOST / CLIENT_ID / CLIENT_SECRET that the Apps runtime injects.
w = WorkspaceClient()

st.set_page_config(page_title="Legal RAG Assistant", page_icon="⚖", layout="wide")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def calculate_cost(prompt_tokens, completion_tokens):
    """Estimated LLM cost for one request, using the configured rate assumptions."""
    input_cost = prompt_tokens / 1_000_000 * INPUT_COST_PER_1M_TOKENS
    output_cost = completion_tokens / 1_000_000 * OUTPUT_COST_PER_1M_TOKENS
    return round(input_cost + output_cost, 8)


def run_sql(statement, parameters=None):
    """
    Run a statement on the workspace SQL warehouse.

    execute_statement returns HTTP 200 even when the SQL fails - the failure shows up as
    status.state = FAILED, not as an exception. So the state has to be checked explicitly,
    otherwise a rejected INSERT looks exactly like a successful one.
    """
    response = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=statement,
        parameters=parameters,
        wait_timeout="50s",
    )

    # A cold warehouse can still be starting when the wait times out, which comes back as
    # PENDING. Poll until the statement reaches a final state.
    waited_seconds = 0
    while response.status.state.value in ("PENDING", "RUNNING") and waited_seconds < 180:
        time.sleep(3)
        waited_seconds += 3
        response = w.statement_execution.get_statement(response.statement_id)

    state = response.status.state.value

    if state != "SUCCEEDED":
        error_message = ""
        if response.status.error is not None:
            error_message = response.status.error.message
        raise RuntimeError(f"SQL {state}: {error_message or 'no error message returned'}")

    return response


def query_to_dataframe(statement):
    """Run a SELECT and return the rows as a DataFrame."""
    response = run_sql(statement)
    if response.result is None or not response.result.data_array:
        return pd.DataFrame()
    columns = [c.name for c in response.manifest.schema.columns]
    return pd.DataFrame(response.result.data_array, columns=columns)


def ask_rag(question, top_k):
    """Send a question to the RAG serving endpoint and return the prediction."""
    response = w.serving_endpoints.query(
        name=ENDPOINT_NAME,
        dataframe_records=[{"question": question, "top_k": top_k}],
    )
    return response.predictions[0]


def save_usage(request_id, question, prediction, cost):
    """Write one usage row. User text is passed as a parameter, not concatenated."""
    # Columns are named explicitly so the insert does not depend on column order,
    # and fails loudly rather than silently if the table schema differs.
    run_sql(
        f"""
        INSERT INTO {USAGE_TABLE}
            (request_id, timestamp, question, prompt_tokens, completion_tokens,
             total_tokens, estimated_cost, model_endpoint, top_k)
        VALUES (:request_id, current_timestamp(), :question, :prompt_tokens,
                :completion_tokens, :total_tokens, :estimated_cost, :model_endpoint, :top_k)
        """,
        parameters=[
            StatementParameterListItem("request_id", value=request_id),
            StatementParameterListItem("question", value=question),
            StatementParameterListItem("prompt_tokens",
                                       value=str(prediction["prompt_tokens"]), type="INT"),
            StatementParameterListItem("completion_tokens",
                                       value=str(prediction["completion_tokens"]), type="INT"),
            StatementParameterListItem("total_tokens",
                                       value=str(prediction["total_tokens"]), type="INT"),
            StatementParameterListItem("estimated_cost", value=str(cost), type="DOUBLE"),
            StatementParameterListItem("model_endpoint", value=prediction["model_endpoint"]),
            StatementParameterListItem("top_k", value=str(prediction["top_k"]), type="INT"),
        ],
    )


def save_feedback(request_id, question, answer, feedback, comment):
    """Write one feedback row, linked to the request by request_id."""
    run_sql(
        f"""
        INSERT INTO {FEEDBACK_TABLE}
            (request_id, timestamp, question, answer, feedback, comment)
        VALUES (:request_id, current_timestamp(), :question, :answer, :feedback, :comment)
        """,
        parameters=[
            StatementParameterListItem("request_id", value=request_id),
            StatementParameterListItem("question", value=question),
            StatementParameterListItem("answer", value=answer),
            StatementParameterListItem("feedback", value=feedback),
            StatementParameterListItem("comment", value=comment or ""),
        ],
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("⚖ Legal RAG Assistant")
st.caption("U.S. Supreme Court opinions and SEC 10-K compliance filings")

if not WAREHOUSE_ID:
    st.warning(
        "No SQL warehouse configured. Answers will work, but usage and feedback cannot be "
        "saved. Add a SQL warehouse resource with the key `sql_warehouse` and redeploy."
    )

with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, 3)
    st.divider()
    st.caption(f"Endpoint: `{ENDPOINT_NAME}`")
    if INPUT_COST_PER_1M_TOKENS == 0 and OUTPUT_COST_PER_1M_TOKENS == 0:
        st.caption("Cost rates not configured, so costs show as $0. Token counts are real.")

    # Checks the whole write path and shows the real error if a grant is missing.
    with st.expander("Diagnostics"):
        st.caption(f"Warehouse ID: `{WAREHOUSE_ID or 'not set'}`")

        if st.button("Test database access"):
            try:
                who = query_to_dataframe("SELECT current_user() AS user")
                st.success(f"Connected as: {who.iloc[0]['user']}")
            except Exception as e:
                st.error(f"Cannot query the warehouse: {e}")

            for table in [USAGE_TABLE, FEEDBACK_TABLE]:
                try:
                    count = query_to_dataframe(f"SELECT COUNT(*) AS n FROM {table}")
                    st.success(f"Read {table}: {count.iloc[0]['n']} rows")

                    # Show the real schema - a table left over from an earlier version
                    # will have different columns and every insert will fail.
                    schema = query_to_dataframe(f"DESCRIBE TABLE {table}")
                    st.caption(f"{table} columns: "
                               + ", ".join(schema["col_name"].tolist()))
                except Exception as e:
                    st.error(f"Cannot read {table}: {e}")

            # A write probe, immediately removed, to prove MODIFY actually works.
            try:
                probe_id = f"diagnostic-{uuid.uuid4()}"
                run_sql(
                    f"INSERT INTO {FEEDBACK_TABLE} "
                    f"(request_id, timestamp, question, answer, feedback, comment) "
                    f"VALUES (:id, current_timestamp(), 'diagnostic', 'diagnostic', "
                    f"'Positive', 'diagnostic probe')",
                    parameters=[StatementParameterListItem("id", value=probe_id)],
                )
                run_sql(
                    f"DELETE FROM {FEEDBACK_TABLE} WHERE request_id = :id",
                    parameters=[StatementParameterListItem("id", value=probe_id)],
                )
                st.success(f"Write test passed on {FEEDBACK_TABLE}")
            except Exception as e:
                st.error(f"Write test failed: {e}")

chat_tab, feedback_tab, usage_tab = st.tabs(["RAG Chat", "Feedback", "Usage & Cost"])

# ---------------------------------------------------------------------------
# Tab 1 - RAG Chat
# ---------------------------------------------------------------------------
with chat_tab:
    question = st.text_area(
        "Question",
        value="What was the central legal issue in this case?",
        height=90,
    )

    if st.button("Submit", type="primary"):
        if not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating..."):
                try:
                    prediction = ask_rag(question.strip(), top_k)
                    cost = calculate_cost(prediction["prompt_tokens"],
                                          prediction["completion_tokens"])
                    request_id = str(uuid.uuid4())

                    # Keep it for the Feedback tab
                    st.session_state["request_id"] = request_id
                    st.session_state["question"] = question.strip()
                    st.session_state["prediction"] = prediction
                    st.session_state["cost"] = cost
                    st.session_state["feedback_done"] = False

                    if WAREHOUSE_ID:
                        try:
                            save_usage(request_id, question.strip(), prediction, cost)
                        except Exception as e:
                            st.warning(f"Answer generated, but usage was not saved: {e}")

                except Exception as e:
                    st.error(f"Could not reach the endpoint: {e}")

    prediction = st.session_state.get("prediction")

    if prediction:
        st.subheader("Answer")
        st.write(prediction["answer"])

        st.subheader("Token usage")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prompt tokens", f"{prediction['prompt_tokens']:,}")
        c2.metric("Completion tokens", f"{prediction['completion_tokens']:,}")
        c3.metric("Total tokens", f"{prediction['total_tokens']:,}")
        c4.metric("Estimated LLM cost", f"${st.session_state['cost']:.6f}")
        st.caption(f"Token counts are {prediction['token_source']}.")

        sources = json.loads(prediction["sources"])
        with st.expander(f"Retrieved sources ({len(sources)})"):
            for i, s in enumerate(sources, start=1):
                label = "Case" if s["source"] == "legal" else "Document"
                st.markdown(
                    f"**{i}. `{s['source']}` — {label} {s['document_id']}, "
                    f"chunk {s['chunk_id']}**"
                )
                st.text(s["text"])

# ---------------------------------------------------------------------------
# Tab 2 - Feedback
# ---------------------------------------------------------------------------
with feedback_tab:
    st.subheader("Rate the last answer")

    if not st.session_state.get("prediction"):
        st.info("Ask a question in the RAG Chat tab first.")
    elif st.session_state.get("feedback_done"):
        st.success("Feedback saved. Thank you.")
    else:
        st.caption(f"Question: {st.session_state['question']}")
        st.text(st.session_state["prediction"]["answer"][:400])

        feedback = st.radio("Was this answer useful?", ["Positive", "Negative"], horizontal=True)
        comment = st.text_area("Comment (optional)", height=80)

        if st.button("Submit feedback"):
            if not WAREHOUSE_ID:
                st.error("No SQL warehouse configured - feedback cannot be saved.")
            else:
                try:
                    save_feedback(
                        st.session_state["request_id"],
                        st.session_state["question"],
                        st.session_state["prediction"]["answer"],
                        feedback,
                        comment,
                    )
                    st.session_state["feedback_done"] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not save feedback: {e}")

# ---------------------------------------------------------------------------
# Tab 3 - Usage & Cost
# ---------------------------------------------------------------------------
with usage_tab:
    st.subheader("Usage and estimated cost")

    if not WAREHOUSE_ID:
        st.info("Configure a SQL warehouse resource to see usage.")
    else:
        try:
            totals = query_to_dataframe(f"""
                SELECT
                    COUNT(*)                                    AS requests,
                    COALESCE(SUM(prompt_tokens), 0)             AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0)         AS completion_tokens,
                    COALESCE(SUM(total_tokens), 0)              AS total_tokens,
                    COALESCE(SUM(estimated_cost), 0)            AS total_cost,
                    COALESCE(AVG(estimated_cost), 0)            AS avg_cost
                FROM {USAGE_TABLE}
            """)

            if totals.empty or int(totals.iloc[0]["requests"]) == 0:
                st.info("No requests recorded yet. Ask a few questions first.")
            else:
                row = totals.iloc[0]
                a, b, c = st.columns(3)
                a.metric("Total requests", f"{int(row['requests']):,}")
                b.metric("Prompt tokens", f"{int(row['prompt_tokens']):,}")
                c.metric("Completion tokens", f"{int(row['completion_tokens']):,}")

                d, e, f = st.columns(3)
                d.metric("Total tokens", f"{int(row['total_tokens']):,}")
                e.metric("Total estimated cost", f"${float(row['total_cost']):.6f}")
                f.metric("Avg cost per request", f"${float(row['avg_cost']):.6f}")

                # Charts over time
                history = query_to_dataframe(f"""
                    SELECT timestamp, total_tokens, estimated_cost
                    FROM {USAGE_TABLE}
                    ORDER BY timestamp
                """)

                if not history.empty:
                    history["timestamp"] = pd.to_datetime(history["timestamp"])
                    history["total_tokens"] = history["total_tokens"].astype(int)
                    history["estimated_cost"] = history["estimated_cost"].astype(float)
                    history = history.set_index("timestamp")

                    st.markdown("**Token usage over time**")
                    st.line_chart(history["total_tokens"])

                    st.markdown("**Estimated cost over time**")
                    st.line_chart(history["estimated_cost"])

                # Feedback counts
                feedback_counts = query_to_dataframe(f"""
                    SELECT feedback, COUNT(*) AS count
                    FROM {FEEDBACK_TABLE}
                    GROUP BY feedback
                """)

                if not feedback_counts.empty:
                    st.markdown("**Feedback**")
                    counts = dict(zip(feedback_counts["feedback"],
                                      feedback_counts["count"].astype(int)))
                    g, h = st.columns(2)
                    g.metric("Positive", counts.get("Positive", 0))
                    h.metric("Negative", counts.get("Negative", 0))

                st.markdown("**Recent requests**")
                st.dataframe(
                    query_to_dataframe(f"""
                        SELECT timestamp, question, total_tokens, estimated_cost, top_k
                        FROM {USAGE_TABLE}
                        ORDER BY timestamp DESC
                        LIMIT 20
                    """),
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Could not read usage data: {e}")
