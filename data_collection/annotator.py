"""
annotator.py
────────────
Step 2 of Phase 1 — Video annotation GUI.

Layout : LEFT  = video canvas + scrubber + transport buttons
         RIGHT = all controls (marks, delete region, label, signer,
                               notes, save, annotation list)

Delete region : frames BETWEEN mark_in and mark_out are removed.
                Everything before IN and everything after OUT is kept.
                Written to a temp file then atomically replaces original.

Output :
  dataset/annotations.json
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import json
import os
import tempfile
import threading
import time
from PIL import Image, ImageTk

# ── Constants ──────────────────────────────────────────────────────────────────
ANNOTATIONS_FILE    = os.path.join("dataset", "annotations.json")
DOWNLOADS_PATH      = os.path.join("dataset", "downloads")
SEQUENCE_LENGTH     = 30
CANVAS_W, CANVAS_H  = 700, 394

# ── Palette ────────────────────────────────────────────────────────────────────
BG       = "#0d0d0d"
SURF     = "#161616"
SURF2    = "#1f1f1f"
SURF3    = "#272727"
BORDER   = "#303030"
TEXT     = "#ececec"
MUTED    = "#5a5a5a"
ACCENT   = "#22c55e"
ACCENT_D = "#15803d"
BLUE_D   = "#1d4ed8"
AMBER    = "#f59e0b"
AMBER_D  = "#b45309"
RED      = "#ef4444"
RED_D    = "#b91c1c"
IN_FG    = "#86efac"
OUT_FG   = "#fca5a5"


# ── Persistence ────────────────────────────────────────────────────────────────
def load_annotations() -> list:
    if os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE) as f:
            return json.load(f)
    return []


def save_annotations(data: list) -> None:
    os.makedirs("dataset", exist_ok=True)
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Widget helpers ─────────────────────────────────────────────────────────────
def _lighten(hex_color: str, amount: int = 28) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#{:02x}{:02x}{:02x}".format(
        min(255, r + amount),
        min(255, g + amount),
        min(255, b + amount),
    )


def mk_btn(parent, text, command, bg, fg,
           font=("Helvetica", 10), padx=6, pady=8,
           width=None, bold=False) -> tk.Button:
    f = (font[0], font[1], "bold") if bold else font
    kw = dict(
        text=text, command=command,
        bg=bg, fg=fg,
        activebackground=_lighten(bg), activeforeground=fg,
        relief="flat", bd=0, highlightthickness=0,
        cursor="hand2", font=f, padx=padx, pady=pady,
    )
    if width:
        kw["width"] = width
    b = tk.Button(parent, **kw)
    normal_bg = bg
    b.bind("<Enter>", lambda _: b.config(bg=_lighten(normal_bg)))
    b.bind("<Leave>", lambda _: b.config(bg=normal_bg))
    return b


def divider(parent, pady=(10, 0)):
    tk.Frame(parent, bg=BORDER, height=1).pack(
        fill="x", padx=12, pady=pady)


def section_label(parent, text: str):
    tk.Label(
        parent, text=text.upper(),
        bg=SURF, fg=MUTED,
        font=("Helvetica", 8, "bold"),
    ).pack(anchor="w", padx=14, pady=(10, 5))


def entry_field(parent, label: str,
                textvariable=None, font_size=11) -> tk.Entry:
    tk.Label(
        parent, text=label,
        bg=SURF, fg=MUTED,
        font=("Helvetica", 9),
    ).pack(anchor="w", padx=14)
    kw = dict(
        bg=SURF3, fg=TEXT,
        insertbackground=TEXT, relief="flat",
        font=("Helvetica", font_size),
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
    )
    if textvariable:
        kw["textvariable"] = textvariable
    e = tk.Entry(parent, **kw)
    e.pack(fill="x", padx=14, ipady=7, pady=(3, 0))
    return e


# ══════════════════════════════════════════════════════════════════════════════
class AnnotatorApp:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NSL Annotator")
        self.root.configure(bg=BG)
        self.root.minsize(1120, 700)

        # video state
        self.cap:           cv2.VideoCapture | None = None
        self.video_path:    str | None = None
        self.total_frames:  int   = 0
        self.fps:           float = 30.0
        self.current_frame: int   = 0
        self.is_playing:    bool  = False
        self._play_thread:  threading.Thread | None = None

        # scrubber recursion guard
        self._scrubbing: bool = False

        # marks
        self.mark_in:  int | None = None
        self.mark_out: int | None = None

        # data
        self.annotations: list = load_annotations()
        self.signer_id = tk.StringVar(value="signer_01")

        self._build_ui()
        self._refresh_table()

    # ══════════════════════════════════════════════════════════════════════════
    #  UI BUILD
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self) -> None:
        self._build_header()
        self._build_body()
        self._build_statusbar()

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self) -> None:
        hdr = tk.Frame(self.root, bg=SURF, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        tk.Label(
            hdr, text="NSL  Annotator",
            bg=SURF, fg=TEXT,
            font=("Helvetica", 13, "bold"),
        ).pack(side="left", padx=16)

        mk_btn(
            hdr, "  Open Video  ", self._open_video,
            bg=SURF3, fg=TEXT, pady=6,
        ).pack(side="left", padx=6, pady=8)

        self.file_label = tk.Label(
            hdr, text="No file loaded",
            bg=SURF, fg=MUTED,
            font=("Helvetica", 10),
        )
        self.file_label.pack(side="left", padx=8)

    # ── Body ──────────────────────────────────────────────────────────────────
    def _build_body(self) -> None:
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)
        self._build_left(body)
        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")
        self._build_right(body)

    # ── Left panel ────────────────────────────────────────────────────────────
    def _build_left(self, parent: tk.Frame) -> None:
        left = tk.Frame(parent, bg=BG)
        left.pack(side="left", fill="both", expand=True,
                  padx=14, pady=12)

        # video canvas
        cf = tk.Frame(left, bg=BORDER, bd=1)
        cf.pack(anchor="center")
        self.canvas = tk.Canvas(
            cf, width=CANVAS_W, height=CANVAS_H,
            bg="#000", highlightthickness=0,
        )
        self.canvas.pack()
        self._draw_placeholder()

        # time row
        tr = tk.Frame(left, bg=BG)
        tr.pack(fill="x", pady=(5, 0))
        self.frame_lbl = tk.Label(
            tr, text="frame 0",
            bg=BG, fg=MUTED, font=("Helvetica", 9))
        self.frame_lbl.pack(side="left")
        self.time_lbl = tk.Label(
            tr, text="0.00 s  /  0.00 s",
            bg=BG, fg=MUTED, font=("Helvetica", 9))
        self.time_lbl.pack(side="right")

        # scrubber
        sty = ttk.Style()
        sty.theme_use("clam")
        sty.configure(
            "S.Horizontal.TScale",
            background=BG, troughcolor=SURF3,
            sliderlength=14, sliderrelief="flat",
        )
        self.scrubber = ttk.Scale(
            left, from_=0, to=100,
            orient="horizontal",
            style="S.Horizontal.TScale",
            command=self._on_scrub,
        )
        self.scrubber.pack(fill="x", pady=(3, 0))

        # region indicator strip
        self.strip = tk.Canvas(left, height=6, bg=BG,
                               highlightthickness=0)
        self.strip.pack(fill="x")

        # ── transport ─────────────────────────────────────────────────────────
        tb = tk.Frame(left, bg=BG)
        tb.pack(pady=(10, 0))

        # Row 1 – frame stepping
        row1 = tk.Frame(tb, bg=BG)
        row1.pack()
        for label, cmd in [
            ("◀◀  -5s", lambda: self._step(-5 * self.fps)),
            ("◀  -1s",  lambda: self._step(-self.fps)),
            ("◀  -1f",  lambda: self._step(-1)),
            ("▶  +1f",  lambda: self._step(1)),
            ("▶  +1s",  lambda: self._step(self.fps)),
            ("▶▶  +5s", lambda: self._step(5 * self.fps)),
        ]:
            mk_btn(
                row1, label, cmd,
                bg=SURF3, fg=TEXT,
                font=("Helvetica", 9), padx=8, pady=6,
            ).pack(side="left", padx=2)

        # Row 2 – play / pause
        row2 = tk.Frame(tb, bg=BG)
        row2.pack(pady=(6, 0))
        self.play_btn = mk_btn(
            row2, "  ▶   Play  ", self._toggle_play,
            bg=ACCENT, fg="#fff",
            font=("Helvetica", 11, "bold"),
            padx=30, pady=8,
        )
        self.play_btn.pack()

        # Row 3 – Mark IN / Mark OUT
        row3 = tk.Frame(tb, bg=BG)
        row3.pack(pady=(8, 0), fill="x")
        mk_btn(
            row3, "  [  Mark IN  ", self._mark_in,
            bg=ACCENT_D, fg=IN_FG,
            font=("Helvetica", 10, "bold"), padx=20, pady=8,
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))
        mk_btn(
            row3, "  Mark OUT  ]  ", self._mark_out,
            bg=RED_D, fg=OUT_FG,
            font=("Helvetica", 10, "bold"), padx=20, pady=8,
        ).pack(side="left", expand=True, fill="x")

        # Row 4 – Delete marked region
        row4 = tk.Frame(tb, bg=BG)
        row4.pack(pady=(6, 0), fill="x")
        mk_btn(
            row4, "  Delete Marked Region  ",
            self._delete_region,
            bg=AMBER_D, fg=AMBER,
            font=("Helvetica", 10, "bold"), padx=20, pady=8,
        ).pack(fill="x")

    # ── Right panel ───────────────────────────────────────────────────────────
    def _build_right(self, parent: tk.Frame) -> None:
        right = tk.Frame(parent, bg=SURF, width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # scrollable interior
        rc = tk.Canvas(right, bg=SURF, highlightthickness=0, bd=0)
        vsb = ttk.Scrollbar(right, orient="vertical", command=rc.yview)
        rc.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        rc.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(rc, bg=SURF)
        win_id = rc.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>",
                   lambda e: rc.configure(scrollregion=rc.bbox("all")))
        rc.bind("<Configure>",
                lambda e: rc.itemconfig(win_id, width=e.width))
        rc.bind_all("<MouseWheel>",
                    lambda e: rc.yview_scroll(
                        int(-1 * (e.delta / 120)), "units"))

        R = inner   # shorthand

        # ── Section 1: mark info ───────────────────────────────────────────────
        section_label(R, "Mark Region")

        badge_row = tk.Frame(R, bg=SURF)
        badge_row.pack(fill="x", padx=14)

        self.in_badge = tk.Label(
            badge_row, text="IN: --",
            bg=SURF2, fg=IN_FG,
            font=("Helvetica", 10),
            anchor="w", padx=8, pady=6)
        self.in_badge.pack(side="left", fill="x",
                           expand=True, padx=(0, 4))

        self.out_badge = tk.Label(
            badge_row, text="OUT: --",
            bg=SURF2, fg=OUT_FG,
            font=("Helvetica", 10),
            anchor="w", padx=8, pady=6)
        self.out_badge.pack(side="left", fill="x", expand=True)

        self.range_badge = tk.Label(
            R, text="No region selected",
            bg=SURF2, fg=MUTED,
            font=("Helvetica", 10),
            anchor="w", padx=8, pady=6)
        self.range_badge.pack(fill="x", padx=14, pady=(4, 0))

        # ── Section 2: annotation fields ──────────────────────────────────────
        divider(R)
        section_label(R, "Annotation")

        self.sign_entry = entry_field(R, "Sign label", font_size=12)
        self.sign_entry.bind("<Return>", lambda _: self._save_annotation())

        tk.Frame(R, bg=SURF, height=6).pack()

        tk.Label(
            R, text="Notes  (optional)",
            bg=SURF, fg=MUTED,
            font=("Helvetica", 9),
        ).pack(anchor="w", padx=14)

        self.notes_text = tk.Text(
            R, height=3,
            bg=SURF3, fg=TEXT,
            insertbackground=TEXT, relief="flat",
            font=("Helvetica", 10),
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            wrap="word",
        )
        self.notes_text.pack(fill="x", padx=14, pady=(3, 0))

        tk.Frame(R, bg=SURF, height=8).pack()

        self.signer_entry = entry_field(
            R, "Signer ID",
            textvariable=self.signer_id,
            font_size=11,
        )

        # ── Section 3: save ───────────────────────────────────────────────────
        divider(R)
        section_label(R, "Save")

        mk_btn(
            R, "  Save Annotation  ",
            self._save_annotation,
            bg=BLUE_D, fg="#93c5fd",
            font=("Helvetica", 11, "bold"),
            padx=20, pady=10,
        ).pack(fill="x", padx=14)

        # ── Section 4: saved annotations list ─────────────────────────────────
        divider(R)
        section_label(R, "Saved Annotations")

        sty2 = ttk.Style()
        sty2.configure(
            "Ann.Treeview",
            background=SURF2, foreground=TEXT,
            fieldbackground=SURF2, rowheight=22,
            font=("Helvetica", 9))
        sty2.configure(
            "Ann.Treeview.Heading",
            background=SURF3, foreground=MUTED,
            relief="flat", font=("Helvetica", 8, "bold"))
        sty2.map("Ann.Treeview",
                 background=[("selected", "#1e3a5f")])

        tw = tk.Frame(R, bg=SURF)
        tw.pack(fill="both", expand=True, padx=14)

        cols   = ("sign", "signer", "start", "end", "frames")
        widths = (88, 84, 52, 52, 46)

        self.tree = ttk.Treeview(
            tw, columns=cols, show="headings",
            style="Ann.Treeview", height=8)
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c.capitalize())
            self.tree.column(c, width=w, anchor="w", stretch=False)

        tv_vsb = ttk.Scrollbar(tw, orient="vertical",
                                command=self.tree.yview)
        self.tree.configure(yscrollcommand=tv_vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tv_vsb.pack(side="right", fill="y")

        self.tree.bind("<Delete>",    self._delete_selected)
        self.tree.bind("<BackSpace>", self._delete_selected)

        mk_btn(
            R, "Delete selected annotation",
            self._delete_selected,
            bg=SURF3, fg=RED,
            font=("Helvetica", 9), pady=6,
        ).pack(fill="x", padx=14, pady=(6, 0))

        tk.Label(
            R, text="Or press Delete / Backspace on a selected row",
            bg=SURF, fg=MUTED,
            font=("Helvetica", 8),
        ).pack(anchor="w", padx=14, pady=(3, 12))

    # ── Status bar ────────────────────────────────────────────────────────────
    def _build_statusbar(self) -> None:
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")
        sb = tk.Frame(self.root, bg=SURF, height=26)
        sb.pack(fill="x")
        sb.pack_propagate(False)

        self.status_lbl = tk.Label(
            sb, text="Ready — open a video to begin",
            bg=SURF, fg=MUTED,
            font=("Helvetica", 9), anchor="w")
        self.status_lbl.pack(side="left", padx=12, fill="y")

        self.count_lbl = tk.Label(
            sb, text="0 annotations",
            bg=SURF, fg=MUTED,
            font=("Helvetica", 9), anchor="e")
        self.count_lbl.pack(side="right", padx=12, fill="y")

    # ══════════════════════════════════════════════════════════════════════════
    #  PLACEHOLDER
    # ══════════════════════════════════════════════════════════════════════════
    def _draw_placeholder(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(
            CANVAS_W // 2, CANVAS_H // 2,
            text="Open a video to begin",
            fill=MUTED, font=("Helvetica", 13))

    # ══════════════════════════════════════════════════════════════════════════
    #  VIDEO LOADING
    # ══════════════════════════════════════════════════════════════════════════
    def _open_video(self) -> None:
        initial = (DOWNLOADS_PATH
                   if os.path.isdir(DOWNLOADS_PATH)
                   else os.path.expanduser("~"))
        path = filedialog.askopenfilename(
            initialdir=initial,
            title="Select video",
            filetypes=[
                ("Video", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All",   "*.*"),
            ],
        )
        if path:
            self._load_video(path)

    def _load_video(self, path: str) -> None:
        # stop any running playback
        if self.is_playing:
            self.is_playing = False
            if self._play_thread and self._play_thread.is_alive():
                self._play_thread.join(timeout=0.5)
        if self.cap:
            self.cap.release()
            self.cap = None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open:\n{path}")
            return

        self.cap           = cap
        self.video_path    = path
        self.fps           = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.mark_in       = None
        self.mark_out      = None

        # update scrubber without firing _on_scrub
        self._scrubbing = True
        self.scrubber.config(to=max(1, self.total_frames - 1))
        self.scrubber.set(0)
        self._scrubbing = False

        self.file_label.config(text=os.path.basename(path), fg=TEXT)
        self._update_marks_ui()
        self._render_frame(0)
        self._status(
            f"{os.path.basename(path)}  |  "
            f"{self.total_frames} frames  |  "
            f"{self.fps:.2f} fps  |  "
            f"{self.total_frames / self.fps:.1f} s"
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  PLAYBACK
    # ══════════════════════════════════════════════════════════════════════════
    def _toggle_play(self) -> None:
        if not self.cap:
            self._status("Open a video first.")
            return
        if self.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        if self.is_playing:
            return
        self.is_playing = True
        self.play_btn.config(text="  ⏸   Pause  ", bg=ACCENT_D)
        self._play_thread = threading.Thread(
            target=self._play_loop, daemon=True)
        self._play_thread.start()

    def _pause(self) -> None:
        self.is_playing = False
        self.play_btn.config(text="  ▶   Play  ", bg=ACCENT)

    def _play_loop(self) -> None:
        interval = 1.0 / self.fps
        while self.is_playing:
            if self.current_frame >= self.total_frames - 1:
                break
            nf = self.current_frame + 1
            self.root.after(0, self._render_frame, nf)
            self.root.after(0, self._set_scrubber_safe, nf)
            time.sleep(interval)
        self.is_playing = False
        self.root.after(
            0, self.play_btn.config,
            {"text": "  ▶   Play  ", "bg": ACCENT},
        )

    def _set_scrubber_safe(self, idx: int) -> None:
        """Set scrubber position without triggering _on_scrub."""
        self._scrubbing = True
        try:
            self.scrubber.set(idx)
        finally:
            self._scrubbing = False

    def _step(self, delta: float) -> None:
        if not self.cap:
            return
        if self.is_playing:
            self._pause()
            time.sleep(0.05)
        target = max(0, min(
            int(self.current_frame + delta),
            self.total_frames - 1))
        self._render_frame(target)
        self._set_scrubber_safe(target)

    def _on_scrub(self, val: str) -> None:
        """Scale callback — blocked during programmatic moves."""
        if self._scrubbing:
            return
        if self.cap and not self.is_playing:
            self._render_frame(int(float(val)))

    def _render_frame(self, idx: int) -> None:
        if not self.cap:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = idx

        h, w = frame.shape[:2]

        # green left bar — inside marked region
        if self.mark_in is not None and idx >= self.mark_in:
            cv2.rectangle(frame, (0, 0), (5, h), (34, 197, 94), -1)
        # red right bar — inside marked region
        if self.mark_out is not None and idx <= self.mark_out:
            cv2.rectangle(frame, (w - 5, 0), (w, h), (239, 68, 68), -1)

        # timecode overlay
        cv2.putText(
            frame,
            f"{idx / self.fps:.2f}s  |  frame {idx}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (200, 200, 200), 1, cv2.LINE_AA,
        )

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb   = cv2.resize(rgb, (CANVAS_W, CANVAS_H),
                           interpolation=cv2.INTER_LINEAR)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.canvas.image = photo
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=photo)

        cur_s   = idx / self.fps
        total_s = self.total_frames / self.fps
        self.time_lbl.config(text=f"{cur_s:.2f} s  /  {total_s:.2f} s")
        self.frame_lbl.config(text=f"frame {idx}")
        self._draw_strip()

    # ── region indicator strip ────────────────────────────────────────────────
    def _draw_strip(self) -> None:
        w = self.strip.winfo_width()
        if w < 4 or self.total_frames < 2:
            return
        self.strip.delete("all")
        self.strip.create_rectangle(0, 0, w, 6, fill=SURF3, outline="")
        if self.mark_in is not None and self.mark_out is not None:
            x1 = int(self.mark_in  / (self.total_frames - 1) * w)
            x2 = int(self.mark_out / (self.total_frames - 1) * w)
            self.strip.create_rectangle(x1, 0, x2, 6,
                                        fill=RED, outline="")
        elif self.mark_in is not None:
            x = int(self.mark_in / (self.total_frames - 1) * w)
            self.strip.create_rectangle(x, 0, x + 3, 6,
                                        fill=IN_FG, outline="")

    # ══════════════════════════════════════════════════════════════════════════
    #  MARKS
    # ══════════════════════════════════════════════════════════════════════════
    def _mark_in(self) -> None:
        if not self.cap:
            self._status("Open a video first.")
            return
        self.mark_in = self.current_frame
        self._update_marks_ui()
        self._render_frame(self.current_frame)
        self._status(
            f"Mark IN  →  frame {self.mark_in}  "
            f"({self.mark_in / self.fps:.2f} s)"
        )

    def _mark_out(self) -> None:
        if not self.cap:
            self._status("Open a video first.")
            return
        self.mark_out = self.current_frame
        self._update_marks_ui()
        self._render_frame(self.current_frame)
        self._status(
            f"Mark OUT  →  frame {self.mark_out}  "
            f"({self.mark_out / self.fps:.2f} s)"
        )

    def _update_marks_ui(self) -> None:
        i_str = (
            f"IN: {self.mark_in / self.fps:.2f}s  (f{self.mark_in})"
            if self.mark_in is not None else "IN: --"
        )
        o_str = (
            f"OUT: {self.mark_out / self.fps:.2f}s  (f{self.mark_out})"
            if self.mark_out is not None else "OUT: --"
        )
        self.in_badge.config(text=i_str)
        self.out_badge.config(text=o_str)

        if self.mark_in is not None and self.mark_out is not None:
            n    = self.mark_out - self.mark_in
            seqs = n // SEQUENCE_LENGTH
            col  = IN_FG if seqs >= 1 else RED
            self.range_badge.config(
                text=(
                    f"{n} frames  ({n / self.fps:.2f}s)"
                    f"  —  {seqs} sequence{'s' if seqs != 1 else ''}"
                    f"  [will be DELETED]"
                ),
                fg=col,
            )
        else:
            self.range_badge.config(
                text="No region selected", fg=MUTED)

        self._draw_strip()

    # ══════════════════════════════════════════════════════════════════════════
    #  DELETE MARKED REGION
    #  Keeps everything BEFORE mark_in and everything AFTER mark_out.
    #  The marked frames in the middle are dropped.
    # ══════════════════════════════════════════════════════════════════════════
    def _delete_region(self) -> None:
        if not self.video_path:
            messagebox.showwarning("No video", "Open a video first.")
            return
        if self.mark_in is None or self.mark_out is None:
            messagebox.showwarning(
                "Missing marks",
                "Set both Mark IN and Mark OUT first.")
            return
        if self.mark_out <= self.mark_in:
            messagebox.showwarning(
                "Invalid range",
                "Mark OUT must be after Mark IN.")
            return

        n_delete = self.mark_out - self.mark_in
        n_keep   = self.total_frames - n_delete

        if not messagebox.askyesno(
            "Confirm Delete Region",
            f"Permanently DELETE frames  {self.mark_in} – {self.mark_out}\n"
            f"  Removing : {n_delete} frames  "
            f"({n_delete / self.fps:.2f} s)\n"
            f"  Keeping  : {n_keep} frames  "
            f"({max(n_keep, 0) / self.fps:.2f} s)\n\n"
            "This overwrites the file and cannot be undone.\n"
            "Continue?",
        ):
            return

        # stop playback and release cap before touching the file
        if self.is_playing:
            self._pause()
            time.sleep(0.1)
        self.cap.release()
        self.cap = None

        self._status("Deleting region — please wait…")
        self.root.update()

        src    = cv2.VideoCapture(self.video_path)
        W      = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
        H      = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        fd, tmp = tempfile.mkstemp(
            suffix=".mp4",
            dir=os.path.dirname(self.video_path))
        os.close(fd)

        writer  = cv2.VideoWriter(tmp, fourcc, self.fps, (W, H))
        written = 0

        # Pass 1 — frames BEFORE mark_in
        src.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(self.mark_in):
            ok, frm = src.read()
            if not ok:
                break
            writer.write(frm)
            written += 1

        # Pass 2 — frames AFTER mark_out
        src.set(cv2.CAP_PROP_POS_FRAMES, self.mark_out)
        while True:
            ok, frm = src.read()
            if not ok:
                break
            writer.write(frm)
            written += 1

        src.release()
        writer.release()

        if written == 0:
            os.remove(tmp)
            messagebox.showerror("Failed", "No frames were written.")
            self._load_video(self.video_path)
            return

        os.replace(tmp, self.video_path)
        size_mb = os.path.getsize(self.video_path) / 1_048_576
        self._status(
            f"Deleted region  |  "
            f"{n_delete} frames removed  |  "
            f"{written} frames kept  |  {size_mb:.1f} MB"
        )
        self.mark_in = self.mark_out = None
        self._load_video(self.video_path)

    # ══════════════════════════════════════════════════════════════════════════
    #  SAVE ANNOTATION
    # ══════════════════════════════════════════════════════════════════════════
    def _save_annotation(self) -> None:
        sign  = self.sign_entry.get().strip().lower().replace(" ", "_")
        notes = self.notes_text.get("1.0", "end").strip()

        if not self.video_path:
            messagebox.showwarning("No video", "Open a video first.")
            return
        if not sign:
            self.sign_entry.focus_set()
            self._status("Enter a sign label before saving.")
            return
        if self.mark_in is None or self.mark_out is None:
            messagebox.showwarning(
                "Missing marks",
                "Set both Mark IN and Mark OUT.")
            return
        if self.mark_out <= self.mark_in:
            messagebox.showwarning(
                "Invalid range",
                "Mark OUT must come after Mark IN.")
            return

        n = self.mark_out - self.mark_in
        if n < SEQUENCE_LENGTH:
            if not messagebox.askyesno(
                "Short clip",
                f"Clip is only {n} frames "
                f"(minimum {SEQUENCE_LENGTH} for one sequence).\n\n"
                "Save anyway?",
            ):
                return

        ann = {
            "video_path":  self.video_path,
            "sign_name":   sign,
            "signer_id":   self.signer_id.get().strip(),
            "notes":       notes,
            "start_sec":   round(self.mark_in  / self.fps, 4),
            "end_sec":     round(self.mark_out / self.fps, 4),
            "start_frame": self.mark_in,
            "end_frame":   self.mark_out,
            "frame_count": n,
            "fps":         round(self.fps, 4),
        }

        self.annotations.append(ann)
        save_annotations(self.annotations)
        self._refresh_table()

        # reset for next annotation
        self.mark_in = self.mark_out = None
        self.sign_entry.delete(0, "end")
        self.notes_text.delete("1.0", "end")
        self._update_marks_ui()
        self._render_frame(self.current_frame)
        self._status(
            f"Saved  '{sign}'   "
            f"{ann['start_sec']}s – {ann['end_sec']}s   "
            f"({n} frames)"
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  DELETE ANNOTATION ROW
    # ══════════════════════════════════════════════════════════════════════════
    def _delete_selected(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            self._status("Select a row in the table first.")
            return
        idx     = self.tree.index(sel[0])
        removed = self.annotations.pop(idx)
        save_annotations(self.annotations)
        self._refresh_table()
        self._status(f"Deleted  '{removed['sign_name']}'")

    # ══════════════════════════════════════════════════════════════════════════
    #  TABLE
    # ══════════════════════════════════════════════════════════════════════════
    def _refresh_table(self) -> None:
        for row in self.tree.get_children():
            self.tree.delete(row)
        for i, a in enumerate(self.annotations):
            tag = "even" if i % 2 == 0 else "odd"
            self.tree.insert("", "end", tags=(tag,), values=(
                a.get("sign_name",   ""),
                a.get("signer_id",   ""),
                f"{a.get('start_sec', 0):.2f}s",
                f"{a.get('end_sec',   0):.2f}s",
                a.get("frame_count", ""),
            ))
        self.tree.tag_configure("even", background=SURF2)
        self.tree.tag_configure("odd",  background="#1a1a1a")
        c = len(self.annotations)
        self.count_lbl.config(
            text=f"{c} annotation{'s' if c != 1 else ''}")

    # ══════════════════════════════════════════════════════════════════════════
    #  STATUS
    # ══════════════════════════════════════════════════════════════════════════
    def _status(self, msg: str) -> None:
        self.status_lbl.config(text=msg)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    AnnotatorApp(root)
    root.mainloop()