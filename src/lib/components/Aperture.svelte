<script lang="ts">
	import { onMount } from 'svelte';

	let apertureEl: HTMLDivElement;
	let tickRing: HTMLDivElement;

	onMount(() => {
		// Generate tick marks
		for (let i = 0; i < 72; i++) {
			const t = document.createElement('div');
			t.className = 'tick' + (i % 9 === 0 ? ' major' : '');
			t.style.transform = `rotate(${i * 5}deg)`;
			tickRing.appendChild(t);
		}

		// Parallax on mouse move
		const blades = apertureEl.querySelectorAll<HTMLElement>('.blade');
		function onMouseMove(e: MouseEvent) {
			const cx = window.innerWidth / 2;
			const cy = window.innerHeight / 2;
			const dx = (e.clientX - cx) / cx;
			const dy = (e.clientY - cy) / cy;
			blades.forEach((b, i) => {
				const a = i * 45;
				const o = 60 + Math.sin((a * Math.PI) / 180) * (dx * 3 + dy * 2) * 3;
				b.style.transform = `translate(-50%,-50%) rotate(${a}deg) translateX(${o}px)`;
			});
		}

		document.addEventListener('mousemove', onMouseMove);
		return () => document.removeEventListener('mousemove', onMouseMove);
	});
</script>

<div class="aperture" bind:this={apertureEl}>
	<div class="aperture-ring" bind:this={tickRing}></div>
	<div class="aperture-ring aperture-ring-outer"></div>
	{#each Array(8) as _, i}
		<div
			class="blade"
			style="transform:translate(-50%,-50%) rotate({i * 45}deg) translateX(60px)"
		></div>
	{/each}
</div>

<style>
	.aperture {
		position: relative;
		width: 420px;
		height: 420px;
		animation: apertureRotate 60s linear infinite;
	}

	@keyframes apertureRotate {
		to {
			transform: rotate(360deg);
		}
	}

	.blade {
		position: absolute;
		top: 50%;
		left: 50%;
		width: 180px;
		height: 50px;
		transform-origin: 0 50%;
		background: linear-gradient(
			135deg,
			color-mix(in srgb, var(--heading) 3%, transparent),
			color-mix(in srgb, var(--heading) 1%, transparent)
		);
		border: 1px solid color-mix(in srgb, var(--heading) 6%, transparent);
		clip-path: polygon(15% 0%, 100% 10%, 100% 90%, 15% 100%);
		transition: transform 0.3s ease;
	}

	.aperture-ring {
		position: absolute;
		top: 50%;
		left: 50%;
		width: 400px;
		height: 400px;
		transform: translate(-50%, -50%);
		border-radius: 50%;
		border: 1px solid var(--border);
		pointer-events: none;
		transition: border-color 0.5s;
	}

	.aperture-ring-outer {
		width: 440px;
		height: 440px;
		border-color: color-mix(in srgb, var(--heading) 3%, transparent);
	}

	:global(.aperture-ring .tick) {
		position: absolute;
		width: 1px;
		height: 8px;
		background: var(--border-light);
		top: 0;
		left: 50%;
		transform-origin: 0 200px;
		transition: background 0.5s;
	}

	:global(.aperture-ring .tick.major) {
		height: 14px;
		background: var(--text-muted);
	}

	@media (max-width: 768px) {
		.aperture {
			width: 280px;
			height: 280px;
		}

		.blade {
			width: 120px;
			height: 35px;
		}

		.aperture-ring {
			width: 260px;
			height: 260px;
		}

		.aperture-ring-outer {
			width: 290px;
			height: 290px;
		}

		:global(.aperture-ring .tick) {
			transform-origin: 0 130px;
		}
	}
</style>
