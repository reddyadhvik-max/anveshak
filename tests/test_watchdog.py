from anveshak.watchdog import should_restart


def test_should_restart():
    assert should_restart("running", False, False, False, 0)
    assert not should_restart("running", True, False, False, 0)    # runner fine
    assert not should_restart("running", False, True, False, 0)    # trainer still alive: never start a second one
    assert not should_restart("running", False, False, True, 0)    # user asked Stop/Pause
    assert not should_restart("failed", False, False, False, 0)    # a real failure needs a fix
    assert not should_restart("running", False, False, False, 3)   # crash loop
