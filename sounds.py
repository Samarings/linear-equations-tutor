"""
sounds.py
---------
Sound effects, confetti, and micro-animation helpers.
All audio is generated programmatically via the Web Audio API —
no external files or CDN needed.

Usage in app.py:
    from sounds import play_correct, play_wrong, play_hint, play_click

Call the appropriate function right after determining if an answer is
correct / wrong, or when a hint / button is triggered.
"""

import streamlit as st


# ── Shared Audio Context bootstrap ───────────────────────────────────────────
# We inject this once per session so subsequent calls can reuse `_ctx`.

_AUDIO_BOOTSTRAP = """
<script>
if (!window._audioCtx) {
    window._audioCtx = new (window.AudioContext || window.webkitAudioContext)();
}
</script>
"""


def _inject_once(key: str, html: str) -> None:
    """Inject a <script> block only once per Streamlit session."""
    if key not in st.session_state:
        st.session_state[key] = True
        st.components.v1.html(html, height=0)


def _play(script_body: str) -> None:
    """Wrap an audio script body in a self-contained HTML block and render it."""
    html = f"""
{_AUDIO_BOOTSTRAP}
<script>
(function() {{
    var ctx = window._audioCtx;
    if (!ctx) return;
    // Resume context if suspended (browser autoplay policy)
    if (ctx.state === 'suspended') ctx.resume();
    {script_body}
}})();
</script>
"""
    st.components.v1.html(html, height=0)


# ── Individual sound functions ────────────────────────────────────────────────

def play_correct() -> None:
    """
    Pleasant two-note 'ding-ding' chime for a correct answer.
    Also triggers the confetti burst + green pulse animation.
    """
    _play("""
    var t = ctx.currentTime;
    [523.25, 659.25, 783.99].forEach(function(freq, i) {
        var osc = ctx.createOscillator();
        var gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, t + i * 0.13);
        gain.gain.setValueAtTime(0, t + i * 0.13);
        gain.gain.linearRampToValueAtTime(0.18, t + i * 0.13 + 0.04);
        gain.gain.exponentialRampToValueAtTime(0.001, t + i * 0.13 + 0.45);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(t + i * 0.13);
        osc.stop(t + i * 0.13 + 0.5);
    });
    """)
    _confetti()
    _pulse_correct()


def play_wrong() -> None:
    """Low 'thud' tone for an incorrect answer + shake animation."""
    _play("""
    var t = ctx.currentTime;
    var osc = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(160, t);
    osc.frequency.exponentialRampToValueAtTime(90, t + 0.25);
    gain.gain.setValueAtTime(0.15, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.3);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 0.35);
    """)
    _shake_wrong()


def play_hint() -> None:
    """Soft ascending 'whoosh' tick when a hint is revealed."""
    _play("""
    var t = ctx.currentTime;
    var osc = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(320, t);
    osc.frequency.linearRampToValueAtTime(520, t + 0.18);
    gain.gain.setValueAtTime(0.10, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.22);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 0.25);
    """)


def play_click() -> None:
    """Subtle short click for button interactions."""
    _play("""
    var t = ctx.currentTime;
    var osc = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(900, t);
    gain.gain.setValueAtTime(0.06, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.06);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 0.08);
    """)


def play_new_problem() -> None:
    """Short rising two-note cue when a new problem is loaded."""
    _play("""
    var t = ctx.currentTime;
    [[400,0],[550,0.1]].forEach(function(p) {
        var osc = ctx.createOscillator();
        var gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(p[0], t + p[1]);
        gain.gain.setValueAtTime(0.09, t + p[1]);
        gain.gain.exponentialRampToValueAtTime(0.001, t + p[1] + 0.18);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(t + p[1]);
        osc.stop(t + p[1] + 0.22);
    });
    """)


# ── Confetti ──────────────────────────────────────────────────────────────────

def _confetti() -> None:
    """Spawn a brief confetti particle burst (pure Canvas, no CDN)."""
    html = """
<canvas id="confetti-canvas" style="
    position:fixed;top:0;left:0;width:100%;height:100%;
    pointer-events:none;z-index:9999;"></canvas>
<script>
(function(){
    var cvs = document.getElementById('confetti-canvas');
    if (!cvs) return;
    var ctx2 = cvs.getContext('2d');
    cvs.width = window.innerWidth;
    cvs.height = window.innerHeight;

    var colors = ['#4CAF82','#86A873','#378ADD','#E8825A','#9B6DD6','#F5C430','#E05C7A'];
    var particles = [];
    for (var i = 0; i < 110; i++) {
        particles.push({
            x: Math.random() * cvs.width,
            y: Math.random() * cvs.height * 0.4 - 20,
            r: Math.random() * 7 + 4,
            d: Math.random() * 80 + 20,
            color: colors[Math.floor(Math.random() * colors.length)],
            tilt: Math.random() * 10 - 10,
            tiltAngle: 0,
            tiltSpeed: Math.random() * 0.07 + 0.05,
            vy: Math.random() * 3 + 2,
            vx: (Math.random() - 0.5) * 2.5,
            opacity: 1
        });
    }

    var frame;
    var tick = 0;
    function draw() {
        ctx2.clearRect(0, 0, cvs.width, cvs.height);
        tick++;
        particles.forEach(function(p) {
            p.tiltAngle += p.tiltSpeed;
            p.y += p.vy;
            p.x += p.vx;
            if (tick > 60) p.opacity -= 0.018;
            ctx2.globalAlpha = Math.max(0, p.opacity);
            ctx2.beginPath();
            ctx2.ellipse(p.x, p.y, p.r, p.r * 0.5, p.tiltAngle, 0, 2 * Math.PI);
            ctx2.fillStyle = p.color;
            ctx2.fill();
        });
        ctx2.globalAlpha = 1;
        particles = particles.filter(function(p){ return p.opacity > 0; });
        if (particles.length > 0) {
            frame = requestAnimationFrame(draw);
        } else {
            ctx2.clearRect(0, 0, cvs.width, cvs.height);
        }
    }
    draw();
})();
</script>
"""
    st.components.v1.html(html, height=0)


# ── CSS animation triggers ────────────────────────────────────────────────────

def _pulse_correct() -> None:
    """Flash a green pulse on the main content area."""
    st.components.v1.html("""
<script>
(function(){
    var el = document.querySelector('section.main');
    if (!el) return;
    el.style.animation = 'none';
    el.offsetHeight; // reflow
    el.style.animation = 'correctPulse 0.7s ease';
    setTimeout(function(){ el.style.animation = ''; }, 750);
})();
</script>
""", height=0)


def _shake_wrong() -> None:
    """Shake the answer input area on wrong answer."""
    st.components.v1.html("""
<script>
(function(){
    var el = document.querySelector('input[type="text"]');
    if (!el) el = document.querySelector('section.main');
    if (!el) return;
    el.style.animation = 'none';
    el.offsetHeight;
    el.style.animation = 'wrongShake 0.45s ease';
    setTimeout(function(){ el.style.animation = ''; }, 500);
})();
</script>
""", height=0)
