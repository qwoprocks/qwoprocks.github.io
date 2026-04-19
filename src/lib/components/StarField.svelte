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

		interface Cloud {
			cx: number;
			cy: number;
			blobs: { ox: number; oy: number; rx: number; ry: number }[];
			speed: number;
			alpha: number;
		}

		let stars: Star[] = [];
		let comets: Comet[] = [];
		let clouds: Cloud[] = [];
		let nextCometTime = 6 + Math.random() * 12;

		// --- Init ---
		function resize() {
			w = canvas.parentElement!.offsetWidth;
			h = canvas.parentElement!.offsetHeight;
			canvas.width = w;
			canvas.height = h;
			initStars();
			initClouds();
		}

		function initStars() {
			const count = Math.floor((w * h) / 4000);
			stars = [];
			for (let i = 0; i < count; i++) {
				const active = Math.random() < 0.2;
				stars.push({
					x: Math.random() * w,
					y: Math.random() * h,
					r: Math.random() * 1.2 + 0.3,
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

		function initClouds() {
			clouds = [];
			const count = 4 + Math.floor(Math.random() * 3); // 4-6 clouds
			for (let i = 0; i < count; i++) {
				const blobCount = 3 + Math.floor(Math.random() * 3);
				const blobs = [];
				for (let j = 0; j < blobCount; j++) {
					blobs.push({
						ox: (Math.random() - 0.5) * 80,
						oy: (Math.random() - 0.5) * 25,
						rx: 25 + Math.random() * 45,
						ry: 12 + Math.random() * 20
					});
				}
				clouds.push({
					cx: Math.random() * w,
					cy: 50 + Math.random() * (h - 120),
					blobs,
					speed: 4 + Math.random() * 8,
					alpha: 0.15 + Math.random() * 0.15
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

		// --- Clouds (light mode) — abstract ellipse outlines ---
		function drawClouds(t: number) {
			const strokeColor =
				getComputedStyle(document.documentElement).getPropertyValue('--text-muted').trim() ||
				'#9a9590';

			for (const cloud of clouds) {
				const offsetX = (cloud.speed * t) % (w + 200) - 100;
				const cx = ((cloud.cx + offsetX) % (w + 200)) - 100;

				ctx.save();
				ctx.strokeStyle = strokeColor;
				ctx.lineWidth = 1;
				ctx.globalAlpha = cloud.alpha;
				for (const blob of cloud.blobs) {
					const bx = cx + blob.ox;
					const by = cloud.cy + blob.oy + Math.sin(t * 0.3 + blob.ox) * 3;
					ctx.beginPath();
					ctx.ellipse(bx, by, blob.rx, blob.ry, 0, 0, Math.PI * 2);
					ctx.stroke();
				}
				ctx.restore();
			}
		}

		// --- Main loop ---
		function draw(timestamp: number) {
			const t = timestamp / 1000;
			ctx.clearRect(0, 0, w, h);

			decayVelocity();
			const theme = getTheme();
			if (theme === 'dark') {
				drawStars(t);
				if (t > nextCometTime) {
					spawnComet(timestamp);
					nextCometTime = t + 6 + Math.random() * 12;
				}
				drawComets(t);
			} else {
				drawClouds(t);
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

		// Decay scroll velocity each frame so effect fades when scrolling stops
		function decayVelocity() {
			scrollVelocity *= 0.92; // smooth exponential decay
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
