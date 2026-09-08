from src.events.journey_tracker import JourneyTracker


def test_journey_tracker_clean_crossing():
    jt = JourneyTracker()

    # Track 1 starts in A
    e1 = jt.update(track_id=1, current_zone="A", timestamp_s=1.0)
    assert e1 is None

    # Enters unzoned space
    e2 = jt.update(track_id=1, current_zone=None, timestamp_s=2.0)
    assert e2 is None

    # Enters B -> triggers A_TO_B
    e3 = jt.update(track_id=1, current_zone="B", timestamp_s=3.0)
    assert e3 is not None
    assert e3["event_type"] == "A_TO_B"
    assert e3["source_zone"] == "A"
    assert e3["destination_zone"] == "B"

    # Staying in B should not trigger duplicate
    e4 = jt.update(track_id=1, current_zone="B", timestamp_s=4.0)
    assert e4 is None


def test_journey_tracker_intermediate_queue():
    jt = JourneyTracker()

    # Track 2 starts in A, goes through QUEUE, arrives at B
    jt.update(track_id=2, current_zone="A", timestamp_s=1.0)
    jt.update(track_id=2, current_zone="QUEUE", timestamp_s=3.0)
    jt.update(track_id=2, current_zone="QUEUE", timestamp_s=8.0)
    e = jt.update(track_id=2, current_zone="B", timestamp_s=10.0)

    assert e is not None
    assert e["event_type"] == "A_TO_B"


def test_journey_tracker_reversal_resilience():
    jt = JourneyTracker()

    # Track 3 starts in B, moves left, reverses back to B, then finally reaches A
    jt.update(track_id=3, current_zone="B", timestamp_s=1.0)
    jt.update(track_id=3, current_zone=None, timestamp_s=2.0)
    jt.update(track_id=3, current_zone="B", timestamp_s=3.0)  # Reversed back to B
    jt.update(track_id=3, current_zone=None, timestamp_s=5.0)
    e = jt.update(track_id=3, current_zone="A", timestamp_s=8.0)  # Finally reaches A

    assert e is not None
    assert e["event_type"] == "B_TO_A"
