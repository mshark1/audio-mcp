"""audio_check_game_project against a tiny synthesized Godot 4 project whose problems are planted by construction."""

import struct

import numpy as np
import pytest
import soundfile as sf

from audio_mcp import godot, loudness, server
from audio_mcp.audio_io import AudioError, write_chunked

SR = 44100

WAV_IMPORT = """[remap]

importer="wav"
type="AudioStreamWAV"
uid="uid://{uid}"
path="res://.godot/imported/{name}-0123456789abcdef.sample"

[deps]

source_file="res://{rel}"
dest_files=["res://.godot/imported/{name}-0123456789abcdef.sample"]

[params]

force/8_bit=false
force/mono=false
force/max_rate=false
force/max_rate_hz=44100
edit/trim=false
edit/normalize=false
edit/loop_mode={loop_mode}
edit/loop_begin=0
edit/loop_end=-1
compress/mode=2
"""

OGG_IMPORT = """[remap]

importer="oggvorbisstr"
type="AudioStreamOggVorbis"
uid="uid://{uid}"
path="res://.godot/imported/{name}-0123456789abcdef.oggvorbisstr"

[deps]

source_file="res://{rel}"
dest_files=["res://.godot/imported/{name}-0123456789abcdef.oggvorbisstr"]

[params]

loop={loop}
loop_offset=0
bpm=0
beat_count=0
bar_beats=4
"""


def tone(seconds, hz=440.0, db=-20.0, fade=0.005, lead=0.0):
    t = np.arange(int(seconds * SR)) / SR
    x = 10 ** (db / 20) * np.sqrt(2) * np.sin(2 * np.pi * hz * t)
    n = int(fade * SR)
    if n:
        ramp = np.sin(0.5 * np.pi * np.linspace(0, 1, n)) ** 2
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return np.concatenate([np.zeros(int(lead * SR)), x])


def put(root, rel, x, loop=None, uid=None, fmt="wav"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    write_chunked(p, x, SR, "VORBIS" if fmt == "ogg" else "PCM_16")  # sf.write of long OGG crashes on Windows
    if loop is not None:
        name = p.name
        uid = uid or f"b{abs(hash(rel)) % 10**12:012d}"
        if fmt == "ogg":
            text = OGG_IMPORT.format(uid=uid, name=name, rel=rel, loop=str(loop).lower())
        else:
            text = WAV_IMPORT.format(uid=uid, name=name, rel=rel, loop_mode=2 if loop else 1)
        (p.parent / (name + ".import")).write_text(text, newline="\n")
    return p


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "mygame"
    root.mkdir()
    (root / "project.godot").write_text('config_version=5\n\n[application]\n\nconfig/name="My Game"\n')
    for name, hz in (("jump", 440), ("coin", 660), ("hit", 330)):
        put(root, f"audio/sfx/{name}.wav", tone(0.4, hz), loop=False)
    put(root, "audio/sfx/step.wav", tone(0.4, 550))  # never imported: no .import
    put(root, "audio/sfx/boom.wav", tone(0.4, 220, db=-10.0), loop=False)  # 10 dB louder
    put(root, "audio/sfx/late.wav", tone(0.4, 500, lead=0.3), loop=False)  # 300 ms leading silence
    for name, hz in (("click", 1000), ("hover", 1200)):
        put(root, f"audio/ui/{name}.wav", tone(0.15, hz, db=-24.0), loop=False)
    put(root, "audio/ui/select.wav", tone(0.15, 900, db=-24.0), loop=True)  # a UI sound that loops
    # 220 Hz x 2 s = 440 whole cycles: a seamless loop. 233 Hz ends mid-cycle: the seam clicks.
    put(root, "audio/music/calm.wav", tone(2.0, 220, db=-26.0, fade=0), loop=True, uid="calmuid")
    put(root, "audio/music/clicky.wav", tone(2.0, 233.3, db=-26.0, fade=0), loop=True)
    put(root, "audio/music/battle.ogg", tone(12.0, 220, db=-26.0, fade=0.5), loop=False, fmt="ogg")
    (root / "scenes").mkdir()
    (root / "scenes" / "level.tscn").write_text(
        '[gd_scene load_steps=4 format=3 uid="uid://scene1"]\n\n'
        '[ext_resource type="AudioStream" uid="uid://bjump" path="res://audio/sfx/jump.wav" id="1_a"]\n'
        '[ext_resource type="AudioStream" uid="uid://calmuid" path="res://audio/music/calm.wav" id="2_b"]\n'
        '[ext_resource type="AudioStream" path="res://audio/sfx/missing_explosion.wav" id="3_c"]\n\n'
        '[node name="Level" type="Node2D"]\n'
    )
    (root / "scripts").mkdir()
    (root / "scripts" / "player.gd").write_text(
        "extends Node\n"
        'var coin = preload("res://audio/Sfx/coin.wav")\n'  # wrong case: breaks in exports
        '# var old = load("res://audio/sfx/old.wav")\n'  # commented out: not a reference
        'func ui(n): return load("res://audio/ui/%s.wav" % n)\n'  # dynamic pattern
        'var boom = load("res://audio/sfx/boom.wav")\n'
        'var late = load("res://audio/sfx/late.wav")\n'
        'var music = ["res://audio/music/clicky.wav", "res://audio/music/battle.ogg"]\n'
    )
    (root / ".godot" / "imported").mkdir(parents=True)
    sf.write(str(root / ".godot" / "imported" / "cache.wav"), np.zeros(100), SR)  # hidden: must be skipped
    return root


def by_file(rep, res):
    return [i for i in rep["issues"] if i["file"] == res or (isinstance(i["file"], list) and res in i["file"])]


def test_planted_problems_found(project):
    rep = server.audio_check_game_project(str(project))
    issues = rep["issues"]

    def one(res, severity, text, category):
        hits = [i for i in by_file(rep, res) if text in i["problem"]]
        assert hits, f"no '{text}' for {res}: {by_file(rep, res)}"
        assert hits[0]["severity"] == severity and hits[0]["category"] == category, hits[0]
        return hits[0]

    loud = one("res://audio/sfx/boom.wav", "warning", "louder than the other sfx", "sfx")
    assert 9.0 < float(loud["problem"].split(" dB")[0]) < 11.0
    one("res://audio/sfx/late.wav", "warning", "300 ms of silence", "sfx")
    one("res://audio/music/clicky.wav", "warning", "Loop seam clicks", "music")
    one("res://audio/music/battle.ogg", "warning", "doesn't loop", "music")
    one("res://audio/ui/select.wav", "warning", "Loops in its import settings", "ui")
    one("res://audio/sfx/missing_explosion.wav", "problem", "Broken reference", "sfx")
    one("res://audio/Sfx/coin.wav", "problem", "wrong letter case", "sfx")
    one("res://audio/sfx/step.wav", "info", "no .import", None)
    assert [i["severity"] for i in issues] == sorted(
        (i["severity"] for i in issues), key=godot.SEVERITY_ORDER.__getitem__
    )
    assert rep["summary"][0].startswith("2 problems, 5 warnings")

    # Clean files: no problems or warnings; the hidden .godot folder and commented-out code are ignored.
    for clean in ("sfx/jump.wav", "sfx/coin.wav", "sfx/hit.wav", "sfx/step.wav", "ui/click.wav", "ui/hover.wav"):
        assert not [i for i in by_file(rep, f"res://audio/{clean}") if i["severity"] != "info"], clean
    assert not [i for i in by_file(rep, "res://audio/music/calm.wav") if i["severity"] != "info"]
    assert not any("old.wav" in str(i["file"]) or "cache.wav" in str(i["file"]) for i in issues)
    assert len(rep["inventory"]) == 12

    refs = rep["references"]
    assert refs["broken"] == ["res://audio/sfx/missing_explosion.wav"]
    # hit and step aren't referenced; coin only through a wrong-case path; ui/* via the "%s" pattern.
    assert set(refs["unreferenced"]) == {
        "res://audio/sfx/coin.wav",
        "res://audio/sfx/hit.wav",
        "res://audio/sfx/step.wav",
    }

    cats = rep["categories"]
    assert cats["sfx"]["files"] == 6 and cats["ui"]["files"] == 3 and cats["music"]["files"] == 3
    assert cats["sfx"]["measure"] == "LUFS momentary max" and cats["music"]["measure"] == "LUFS integrated"
    assert abs(cats["sfx"]["median"] - (-20.0)) < 1.0
    assert cats["music"]["looping"] == 2 and cats["ui"]["looping"] == 1
    assert "suggested_batch" in cats["sfx"] and "suggested_batch" not in cats["ui"]


def test_suggested_loudness_fixes_work(project, tmp_path):
    rep = godot.check_project(str(project))
    # Per-file fix for the outlier, applied in place, then re-check: the outlier is gone.
    fix = next(i for i in by_file(rep, "res://audio/sfx/boom.wav") if "louder" in i["problem"])["fix"]
    boom = str(project / "audio/sfx/boom.wav")
    server.audio_process(boom, fix["audio_process_operations"], output=boom, overwrite=True)
    after = godot.check_project(str(project))
    assert not [i for i in by_file(after, "res://audio/sfx/boom.wav") if "louder" in i["problem"]]

    # The per-category batch suggestion brings every SFX to the category median.
    sug = rep["categories"]["sfx"]["suggested_batch"]["audio_batch_process"]
    assert sug["target"] == str(project / "audio" / "sfx")
    res = server.audio_batch_process(**sug)
    assert res["failed"] == 0 and res["processed"] == 6
    target = sug["operations"][0]["lufs"]
    for p in (tmp_path / "mygame_audio_fixed" / "sfx").glob("*.wav"):
        x, sr = sf.read(str(p), always_2d=True)
        assert abs(loudness.momentary_max(x, sr) - target) < 0.6, p.name


def test_detect_from_wav_smpl_loop_and_category_override(project):
    # loop_mode=0 ("Detect From WAV") loops only if the file carries a smpl loop: add one to battle-like music.
    p = put(project, "audio/music/smpl.wav", tone(2.0, 220, db=-26.0, fade=0), loop=True)
    imp = p.with_name(p.name + ".import")
    imp.write_text(imp.read_text().replace("edit/loop_mode=2", "edit/loop_mode=0"), newline="\n")
    assert godot.import_settings(p)["loop"] is False
    data = bytearray(p.read_bytes())
    smpl = struct.pack("<9I", 0, 0, 0, 60, 0, 0, 0, 1, 0) + struct.pack("<6I", 0, 0, 0, 88199, 0, 0)
    data += b"smpl" + struct.pack("<I", len(smpl)) + smpl
    data[4:8] = struct.pack("<I", len(data) - 8)
    p.write_bytes(bytes(data))
    s = godot.import_settings(p)
    assert s["loop"] is True and s["loop_kind"] == "forward" and s["loop_end"] == 88200

    # A scene whose path is stale but whose uid still resolves isn't broken (Godot 4.x falls back to the uid).
    (project / "scenes" / "menu.tscn").write_text(
        '[ext_resource type="AudioStream" uid="uid://calmuid" path="res://audio/old/calm.wav" id="1_a"]\n'
    )
    rep = godot.check_project(str(project), categories={"res://audio/sfx/": "UI", "*/battle.ogg": "ambient"})
    assert [i["severity"] for i in by_file(rep, "res://audio/old/calm.wav")] == ["info"]
    assert "res://audio/old/calm.wav" not in rep["references"]["broken"]
    assert rep["categories"]["ui"]["files"] == 9 and "sfx" not in rep["categories"]
    assert (
        next(i for i in by_file(rep, "res://audio/music/battle.ogg") if "loop" in i["problem"])["category"]
        == "ambience"
    )
    assert not [i for i in by_file(rep, "res://audio/music/smpl.wav") if i["severity"] != "info"]


def test_categorize_and_parse_defensively():
    rules = []
    assert godot.categorize("assets/Music/level1.ogg", rules)[0] == "music"
    assert godot.categorize("sounds/BGM_title.ogg", rules) == ("music", "name")
    assert godot.categorize("audio/Ambience/forest.ogg", rules)[0] == "ambience"
    assert godot.categorize("audio/VO/hero_01.wav", rules)[0] == "voice"
    assert godot.categorize("audio/sound_effects/x.wav", rules)[0] == "sfx"
    assert godot.categorize("audio/misc/thing.wav", rules) == ("uncategorized", "none")
    odd = '[remap]\nimporter="wav"\nvalid=false\nmetadata={\n"key": [1, 2]\n}\n[params]\nedit/loop_mode="weird"\nnew/key=@#$\n'
    sec = godot.parse_import(odd)
    assert sec["remap"]["valid"] is False and sec["params"]["edit/loop_mode"] == "weird"
    with pytest.raises(AudioError, match="Unknown category"):
        godot._caller_rules({"audio": "explosions"})


def test_contract(tmp_path, monkeypatch, project):
    with pytest.raises(AudioError, match="project.godot"):
        server.audio_check_game_project(str(tmp_path))
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "project.godot").write_text("config_version=5\n")
    with pytest.raises(AudioError, match="No audio files"):
        server.audio_check_game_project(str(empty))
    (empty / "project.godot").write_text("config_version=4\n")
    with pytest.raises(AudioError, match="Godot 3"):
        server.audio_check_game_project(str(empty))
    with pytest.raises(AudioError, match="Not a directory"):
        server.audio_check_game_project(str(project / "project.godot"))
    # Odd inputs give a clean report: corrupt audio, garbage .import, binary scene, invalid import.
    (project / "audio/sfx/corrupt.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
    (project / "audio/sfx/corrupt.wav.import").write_bytes(b"\xff\xfe[remap\nimporter=\n=x\n")
    (project / "scenes/bin.tscn").write_bytes(bytes(range(256)) * 10)
    imp = project / "audio/sfx/hit.wav.import"
    imp.write_text(imp.read_text().replace('type="AudioStreamWAV"', 'type="AudioStreamWAV"\nvalid=false'))
    rep = server.audio_check_game_project(str(project), max_files=1)
    assert any("corrupt.wav" in u for u in rep.get("unreadable", [])) or "Analyzed the first 1" in " ".join(
        rep["notes"]
    )
    rep = server.audio_check_game_project(str(project))
    assert any("corrupt.wav" in u for u in rep["unreadable"])
    assert any("valid=false" in i["problem"] for i in by_file(rep, "res://audio/sfx/hit.wav"))
    # Sandbox: a project outside AUDIO_MCP_ALLOWED_DIRS is refused.
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("AUDIO_MCP_ALLOWED_DIRS", str(allowed))
    with pytest.raises(AudioError, match="not allowed"):
        server.audio_check_game_project(str(project))
