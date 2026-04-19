<script lang="ts">
	import { onMount } from 'svelte';

	let canvas: HTMLCanvasElement;

	onMount(() => {
		const ctx = canvas.getContext('2d')!;
		let w = 0;
		let h = 0;
		let animId: number;
		let scrollVelocity = 0;
		let lastScrollY = 0;
		let lastScrollTime = 0;

		// --- Types ---
		interface Star {
			x: number;
			y: number;
			r: number;
			baseAlpha: number;
			s: number; // angular frequency for oscillation
			p: number; // phase offset
			active: boolean; // true = noticeable twinkler (20%)
		}

		interface Comet {
			x: number;
			y: number;
			angle: number;
			speed: number;
			length: number;
			headSize: number;
			startTime: number;
			duration: number;
			active: boolean;
		}

		interface Kana {
			x: number;
			y: number;
			vx: number;
			vy: number;
			char: string;
			size: number;
			alpha: number;
			phase: number;
			freq: number;
			accent: boolean;
		}

		const HIRAGANA = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん';

		let stars: Star[] = [];
		let comets: Comet[] = [];
		let kana: Kana[] = [];
		let nextCometTime = 6 + Math.random() * 12;

		// --- Init ---
		function resize() {
			w = canvas.parentElement!.offsetWidth;
			h = canvas.parentElement!.offsetHeight;
			canvas.width = w;
			canvas.height = h;
			initStars();
			initKana();
		}

		function initStars() {
			const count = Math.floor((w * h) / 4000);
			stars = [];
			for (let i = 0; i < count; i++) {
				const active = Math.random() < 0.2;
				stars.push({
					x: Math.random() * w,
					y: Math.random() * h,
					r: Math.random() * 1.2 + 0.6,
					baseAlpha: Math.random() * 0.4 + 0.15,
					// Active: 2-5s cycle; Steady: 15-40s cycle
					s: active
						? (2 * Math.PI) / (2 + Math.random() * 3)
						: (2 * Math.PI) / (15 + Math.random() * 25),
					p: Math.random() * Math.PI * 2,
					active
				});
			}
		}

		function initKana() {
			kana = [];
			const count = 35 + Math.floor(Math.random() * 15);
			for (let i = 0; i < count; i++) {
				const accent = Math.random() < 0.12;
				kana.push({
					x: Math.random() * w,
					y: Math.random() * h,
					vx: 0, // driven by Brownian motion in drawKana
					vy: -(15 + Math.random() * 25), // px/s upward
					char: HIRAGANA[Math.floor(Math.random() * HIRAGANA.length)],
					size: 12 + Math.random() * 20,
					alpha: accent ? 0.45 + Math.random() * 0.25 : 0.18 + Math.random() * 0.22,
					phase: Math.random() * Math.PI * 2,
					freq: 1 + Math.random() * 2, // Hz for Brownian wobble
					accent
				});
			}
		}

		function getTheme(): string {
			return document.documentElement.getAttribute('data-theme') ?? 'dark';
		}

		// --- Stars ---
		function drawStars(t: number) {
			const textColor = getComputedStyle(document.documentElement)
				.getPropertyValue('--text')
				.trim();
			// Chromatic aberration based on scroll velocity
			const v = Math.min(Math.abs(scrollVelocity) / 1500, 1); // normalize: 1500px/s = full effect
			const maxOffset = 3;
			const aberrationOffset = v * maxOffset;
			const aberrationAlpha = v * 1.2;

			for (const s of stars) {
				// Two combined harmonics for organic oscillation
				const osc1 = Math.sin(t * s.s + s.p);
				const osc2 = Math.sin(t * s.s * 0.6 + s.p * 1.7);
				const osc = osc1 * 0.7 + osc2 * 0.3;

				let alpha: number;
				if (s.active) {
					// Active: vary ~35%, never below 60% of base
					alpha = s.baseAlpha * (0.6 + 0.4 * (osc * 0.5 + 0.5));
				} else {
					// Steady: barely perceptible ~8% variation
					alpha = s.baseAlpha * (0.92 + 0.08 * (osc * 0.5 + 0.5));
				}

				// Chromatic aberration horizontal split on scroll velocity
				if (aberrationOffset > 0.1) {
					ctx.beginPath();
					ctx.arc(s.x + aberrationOffset, s.y, s.r * 1.2, 0, Math.PI * 2);
					ctx.fillStyle = `rgba(255,50,50,${aberrationAlpha * alpha})`;
					ctx.fill();

					ctx.beginPath();
					ctx.arc(s.x - aberrationOffset, s.y, s.r * 1.2, 0, Math.PI * 2);
					ctx.fillStyle = `rgba(50,100,255,${aberrationAlpha * alpha})`;
					ctx.fill();
				}

				// Normal star
				ctx.beginPath();
				ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
				ctx.fillStyle = textColor.startsWith('#')
					? textColor +
						Math.round(alpha * 255)
							.toString(16)
							.padStart(2, '0')
					: `rgba(200,196,192,${alpha})`;
				ctx.fill();
			}
		}

		// --- Comets (simple gradient trail) ---
		function hexToRgb(hex: string) {
			const m = hex.match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
			return m
				? { r: parseInt(m[1], 16), g: parseInt(m[2], 16), b: parseInt(m[3], 16) }
				: null;
		}

		function spawnComet(t: number) {
			let x: number, y: number;
			if (Math.random() < 0.6) {
				x = Math.random() * w;
				y = 0;
			} else {
				x = w;
				y = Math.random() * h * 0.6;
			}
			const angle = ((225 + (Math.random() - 0.5) * 50) * Math.PI) / 180;
			comets.push({
				x,
				y,
				angle,
				speed: 0.15 + Math.random() * 0.25,
				length: 80 + Math.random() * 170,
				headSize: 1 + Math.random() * 2,
				startTime: t,
				duration: 1000 + Math.random() * 1500,
				active: true
			});
		}

		function drawComets(t: number) {
			const accentHex =
				getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
			const rgb = hexToRgb(accentHex) || { r: 255, g: 64, b: 64 };
			const tMs = t * 1000;

			for (let i = comets.length - 1; i >= 0; i--) {
				const c = comets[i];
				const elapsed = tMs - c.startTime;
				const progress = elapsed / c.duration;
				if (progress > 1) {
					comets.splice(i, 1);
					continue;
				}

				const dist = elapsed * c.speed;
				const cx = c.x + Math.cos(c.angle) * dist;
				const cy = c.y - Math.sin(c.angle) * dist;
				const dx = Math.cos(c.angle);
				const dy = -Math.sin(c.angle);

				let opacity = 1;
				if (progress < 0.1) opacity = progress / 0.1;
				else if (progress > 0.7) opacity = 1 - (progress - 0.7) / 0.3;

				// Gradient trail
				const tailX = cx - dx * c.length;
				const tailY = cy - dy * c.length;
				const gradient = ctx.createLinearGradient(tailX, tailY, cx, cy);
				gradient.addColorStop(0, 'rgba(0,0,0,0)');
				gradient.addColorStop(0.7, `rgba(${rgb.r},${rgb.g},${rgb.b},${0.3 * opacity})`);
				gradient.addColorStop(1, `rgba(${rgb.r},${rgb.g},${rgb.b},${0.9 * opacity})`);

				ctx.beginPath();
				ctx.moveTo(tailX, tailY);
				ctx.lineTo(cx, cy);
				ctx.strokeStyle = gradient;
				ctx.lineWidth = c.headSize * 0.6;
				ctx.lineCap = 'round';
				ctx.stroke();

				// Bright head
				ctx.beginPath();
				ctx.arc(cx, cy, c.headSize * 0.5, 0, Math.PI * 2);
				ctx.fillStyle = `rgba(255,255,255,${0.95 * opacity})`;
				ctx.fill();

				// Glow
				ctx.beginPath();
				ctx.arc(cx, cy, c.headSize * 1.5, 0, Math.PI * 2);
				ctx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},${0.3 * opacity})`;
				ctx.fill();
			}
		}

		// --- Kana (light mode) — drifting hiragana characters ---
		function drawKana(t: number, dt: number) {
			const styles = getComputedStyle(document.documentElement);
			const dimColor = styles.getPropertyValue('--text-dim').trim() || '#6a6560';
			const accentColor = styles.getPropertyValue('--accent').trim() || '#d03030';

			// Chromatic aberration from scroll velocity (shared with stars)
			const v = Math.min(Math.abs(scrollVelocity) / 1500, 1);
			const aberrationOffset = v * 3;
			const aberrationAlpha = v * 1.2;

			// vx damping: 0.95/frame at 60fps → per-second decay constant
			const VX_DECAY_RATE = -Math.log(0.95) * 60;

			for (const k of kana) {
				// Smooth sine wander (deterministic, per-symbol via unique phase/freq)
				const wanderX = Math.sin(t * k.freq + k.phase) * 12;

				// Brownian impulse: scale by sqrt(dt) for frame-rate independent diffusion
				// Each symbol gets its own Math.random() call — uncorrelated per symbol
				k.vx += (Math.random() - 0.5) * 60 * Math.sqrt(dt);
				// Frame-rate independent damping
				k.vx *= Math.exp(-VX_DECAY_RATE * dt);

				k.x += (k.vx + wanderX * 0.3) * dt;
				k.y += k.vy * dt;

				// Wrap
				if (k.y < -k.size) { k.y = h + k.size; k.x = Math.random() * w; }
				if (k.x < -k.size * 2) k.x = w + k.size;
				if (k.x > w + k.size * 2) k.x = -k.size;

				const osc = Math.sin(t * 0.8 + k.phase) * 0.15 + 0.85;
				const alpha = k.alpha * osc;

				ctx.save();
				ctx.font = `300 ${k.size}px 'Space Grotesk', sans-serif`;
				ctx.textAlign = 'center';
				ctx.textBaseline = 'middle';

				// Chromatic aberration on scroll (same logic as stars)
				if (aberrationOffset > 0.1) {
					ctx.globalAlpha = aberrationAlpha * alpha;
					ctx.fillStyle = 'rgba(255,50,50,1)';
					ctx.fillText(k.char, k.x + aberrationOffset, k.y);
					ctx.fillStyle = 'rgba(50,100,255,1)';
					ctx.fillText(k.char, k.x - aberrationOffset, k.y);
				}

				// Normal character
				ctx.globalAlpha = alpha;
				ctx.fillStyle = k.accent ? accentColor : dimColor;
				ctx.fillText(k.char, k.x, k.y);
				ctx.restore();
			}
		}

		// --- Main loop ---
		let lastFrameTime = 0;
		function draw(timestamp: number) {
			const t = timestamp / 1000;
			const dt = lastFrameTime ? Math.min((timestamp - lastFrameTime) / 1000, 0.05) : 0.016;
			lastFrameTime = timestamp;
			ctx.clearRect(0, 0, w, h);

			decayVelocity(dt);
			const theme = getTheme();
			if (theme === 'dark') {
				drawStars(t);
				if (t > nextCometTime) {
					spawnComet(timestamp);
					nextCometTime = t + 6 + Math.random() * 12;
				}
				drawComets(t);
			} else {
				drawKana(t, dt);
			}

			animId = requestAnimationFrame(draw);
		}

		// --- Scroll velocity tracking for chromatic aberration ---
		function handleScroll() {
			const now = performance.now();
			const dt = now - lastScrollTime;
			if (dt > 0) {
				const dy = window.scrollY - lastScrollY;
				scrollVelocity = (dy / dt) * 1000; // px per second
			}
			lastScrollY = window.scrollY;
			lastScrollTime = now;
		}

		// Decay scroll velocity — frame-rate independent exponential decay
		// 0.92 per frame at 60fps → half-life ≈ 0.2s
		const SCROLL_DECAY_RATE = -Math.log(0.92) * 60; // per-second decay constant
		function decayVelocity(dt: number) {
			scrollVelocity *= Math.exp(-SCROLL_DECAY_RATE * dt);
			if (Math.abs(scrollVelocity) < 1) scrollVelocity = 0;
		}

		window.addEventListener('resize', resize);
		window.addEventListener('scroll', handleScroll, { passive: true });
		resize();
		animId = requestAnimationFrame(draw);

		return () => {
			cancelAnimationFrame(animId);
			window.removeEventListener('resize', resize);
			window.removeEventListener('scroll', handleScroll);
		};
	});
</script>

<canvas class="star-field" bind:this={canvas}></canvas>

<style>
	.star-field {
		position: absolute;
		inset: 0;
		z-index: 0;
		pointer-events: none;
	}
</style>
