"""
tests/test_voice_agent.py
=========================
Comprehensive tests for smartml.agent (VoiceAgent, NLP normalization,
compound command splitting, partial file matching, disambiguation,
walkthrough sentences, and path boundary security).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from smartml.agent import (
    EnergyVAD,
    VoiceAgent,
    _parse_voice_command,
    find_dataset_candidates,
    find_dataset_file,
    is_within_allowed_dirs,
    normalize_transcript,
    parse_hotkey,
    speakable,
)


@pytest.fixture
def survey_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60
    hours = rng.uniform(1.0, 10.0, size=n)
    gpa = rng.uniform(2.0, 4.0, size=n)
    recommend = ((hours * 0.4 + gpa * 0.8) > 4.2).astype(int)
    return pd.DataFrame({
        "Study_Hours": hours,
        "GPA": gpa,
        "Would_Recommend": recommend,
    })


def test_normalize_transcript_and_compound_split():
    raw = "load student dot csv and then build a model to predict Would Recommend"
    norm = normalize_transcript(raw)
    assert "student.csv" in norm
    cmds = _parse_voice_command(raw)
    assert len(cmds) == 2
    assert cmds[0].startswith("load student.csv")
    assert "predict Would Recommend" in cmds[1]


def test_number_words_conversion():
    norm = normalize_transcript("group my customers into three segments")
    assert "3 segments" in norm


def test_parse_hotkey_formats():
    assert parse_hotkey("Ctrl+Alt+M") == "<ctrl>+<alt>+m"
    assert parse_hotkey("<ctrl>+<shift>+a") == "<ctrl>+<shift>+a"
    assert parse_hotkey("Escape") == "<esc>"
    with pytest.raises(ValueError):
        parse_hotkey("")


def test_speakable_summarises_long_output():
    long_text = "\n".join([f"- Step {i}: processed feature column_{i}" for i in range(30)])
    short = speakable(long_text, max_chars=200)
    assert len(short) <= 205
    assert "Full details are on screen." in short


def test_partial_filename_matching_and_disambiguation(tmp_path: Path, survey_df: pd.DataFrame):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    f1 = allowed / "student_survey.csv"
    survey_df.to_csv(f1, index=False)

    # Single partial match: "load the student file" -> finds student_survey.csv
    found = find_dataset_file("load the student file", search_dirs=[allowed])
    assert found == str(f1.resolve())

    # Add a second matching file -> VoiceAgent asks to disambiguate
    f2 = allowed / "student_grades.csv"
    survey_df.to_csv(f2, index=False)

    cands = find_dataset_candidates("load the student file", search_dirs=[allowed])
    assert len(cands) == 2

    agent = VoiceAgent(
        output_dir=str(tmp_path / "out"),
        search_dirs=[allowed],
        speak_replies=False,
    )
    reply1 = agent.handle_command("Load the student file")
    assert "I found 2 matching files" in reply1
    assert len(agent.pending_candidates) == 2

    # User picks "1"
    reply2 = agent.handle_command("first one")
    assert "Loaded 'student_" in reply2
    assert agent.has_dataset
    assert len(agent.pending_candidates) == 0


def test_path_escape_blocked_in_find_dataset_file(tmp_path: Path, survey_df: pd.DataFrame):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()

    secret_csv = outside / "secret.csv"
    survey_df.to_csv(secret_csv, index=False)

    # 1. Explicit absolute path outside allowed_dirs must return None / []
    assert find_dataset_file(str(secret_csv), search_dirs=[allowed]) is None
    assert find_dataset_candidates(f"load {secret_csv}", search_dirs=[allowed]) == []

    # 2. Relative '..' traversal escaping allowed_dirs must return None / []
    traversal = allowed / ".." / "outside" / "secret.csv"
    assert not is_within_allowed_dirs(traversal, [allowed])
    assert find_dataset_file(str(traversal), search_dirs=[allowed]) is None

    # 3. VoiceAgent refuses explicit outside path
    agent = VoiceAgent(
        output_dir=str(tmp_path / "out"),
        search_dirs=[allowed],
        speak_replies=False,
    )
    reply = agent.handle_command(f"load {secret_csv}")
    assert "Access denied" in reply or "couldn't find" in reply
    assert not agent.has_dataset


def test_walkthrough_sentences_end_to_end(tmp_path: Path, survey_df: pd.DataFrame):
    allowed = tmp_path / "data"
    allowed.mkdir()
    csv_path = allowed / "student_survey.csv"
    survey_df.to_csv(csv_path, index=False)

    agent = VoiceAgent(
        output_dir=str(tmp_path / "out"),
        search_dirs=[allowed],
        speak_replies=False,
    )

    # 1. "Load the student file"
    r_load = agent.handle_command("Load the student file")
    assert "Loaded 'student_survey.csv'" in r_load
    assert agent.has_dataset

    # 2. "Summary of the data" and "Summarise the data"
    r_sum = agent.handle_command("Summary of the data")
    assert "60 rows" in r_sum
    assert "Data quality score" in r_sum
    assert agent.assistant.session.dataset_report is not None

    # 3. "Build a model to predict Would Recommend" (auto-sets target & trains)
    r_train = agent.handle_command("Build a model to predict Would Recommend")
    assert agent.assistant.session.target == "Would_Recommend"
    assert agent.assistant.session.training_result is not None
    assert "Trained and compared" in r_train

    # 4. "Why did that model win?"
    r_why = agent.handle_command("Why did that model win?")
    assert "was selected because" in r_why

    # 5. "What matters most?"
    r_imp = agent.handle_command("What matters most?")
    assert "Feature importance" in r_imp

    # 6. "Remove outliers" when 0 outliers exist preserves the trained model
    df_no_outliers = pd.DataFrame({
        "A": [10.0, 11.0, 12.0, 10.5, 11.5, 10.2, 11.8, 10.9, 11.1, 11.3] * 4,
        "B": [20.0, 21.0, 22.0, 20.5, 21.5, 20.2, 21.8, 20.9, 21.1, 21.3] * 4,
        "Target": [0, 1] * 20,
    })
    agent2 = VoiceAgent(df=df_no_outliers, target="Target", output_dir=str(tmp_path / "out2"), speak_replies=False)
    agent2.handle_command("train the best model")
    assert agent2.assistant.session.training_result is not None
    r_out = agent2.handle_command("Remove outliers")
    assert "Removed 0 outlier" in r_out
    assert agent2.assistant.session.training_result is not None

    # 7. Real data mutation (fill missing) invalidates stale trained model
    df_with_na = df_no_outliers.copy()
    df_with_na.loc[0, "A"] = np.nan
    agent3 = VoiceAgent(df=df_with_na, target="Target", output_dir=str(tmp_path / "out3"), speak_replies=False)
    agent3.handle_command("train the best model")
    assert agent3.assistant.session.training_result is not None
    r_fill = agent3.handle_command("fill missing in A with 99")
    assert "Filled 1 missing" in r_fill
    assert agent3.assistant.session.training_result is None

    # 8. "Group my customers into 3 segments"
    r_clust = agent.handle_command("Group my customers into 3 segments")
    assert "Clustering complete" in r_clust
    assert "Clusters found: 3" in r_clust

    # 9. "Show me the report" opens browser and generates report.html
    with patch("webbrowser.open") as mock_open:
        r_rep = agent.handle_command("Show me the report")
        assert "HTML report generated" in r_rep
        assert mock_open.called


def test_energy_vad_rms():
    silence = np.zeros(1600, dtype=np.int16)
    tone = (np.sin(np.linspace(0, 50, 1600)) * 16000).astype(np.int16)
    assert EnergyVAD.rms(silence) == 0.0
    assert EnergyVAD.rms(tone) > 0.2


def test_find_dataset_depth_limit_and_audit_log_opt_in(tmp_path: Path, survey_df: pd.DataFrame):
    from smartml.agent.audio import SpeechToText

    allowed = tmp_path / "root_data"
    d2 = allowed / "lvl1" / "lvl2"
    d3 = d2 / "lvl3"
    d3.mkdir(parents=True)

    ok_file = d2 / "depth_two_sales.csv"
    too_deep = d3 / "depth_three_sales.csv"
    survey_df.to_csv(ok_file, index=False)
    survey_df.to_csv(too_deep, index=False)

    matches = find_dataset_candidates("sales", search_dirs=[allowed])
    assert matches == [str(ok_file.resolve())]

    # Audit log must be off by default (A5.16)
    out_default = tmp_path / "out_default"
    va_default = VoiceAgent(df=survey_df, output_dir=str(out_default), speak_replies=False)
    va_default.handle_command("Summary of the data")
    assert not (out_default / "agent_log.jsonl").exists()

    # Audit log enabled explicitly writes JSONL
    out_optin = tmp_path / "out_optin"
    va_optin = VoiceAgent(df=survey_df, output_dir=str(out_optin), speak_replies=False, audit_log=True)
    va_optin.handle_command("Summary of the data")
    assert (out_optin / "agent_log.jsonl").is_file()

    # Whisper transcription must pass in-memory BytesIO and never touch NamedTemporaryFile (A5.16)
    stt = SpeechToText(engine="whisper")
    fake_seg = type("Seg", (), {"text": "load student file"})()

    class _FakeWhisper:
        def __init__(self):
            self.received_type = None

        def transcribe(self, audio, **_kwargs):
            self.received_type = type(audio).__name__
            return [fake_seg], None

    fake_model = _FakeWhisper()
    stt._whisper_model = fake_model
    with patch("tempfile.NamedTemporaryFile", side_effect=AssertionError("Audio written to disk")):
        txt = stt.transcribe(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    assert txt == "load student file"
    assert fake_model.received_type == "BytesIO"


