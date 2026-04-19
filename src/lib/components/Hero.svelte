<script lang="ts">
	import StarField from './StarField.svelte';
	import Aperture from './Aperture.svelte';
</script>

<section class="hero" id="hero">
	<StarField />

	<!-- Shuriken sparkle decorations -->
	<span class="sparkle" aria-hidden="true" style="top:16%;right:12%">
		<svg width="16" height="16" viewBox="0 0 24 24" stroke="var(--accent)"
			><path d="M12 2 L14 10 L22 12 L14 14 L12 22 L10 14 L2 12 L10 10 Z" /></svg
		>
	</span>
	<span class="sparkle" aria-hidden="true" style="top:52%;right:20%">
		<svg width="12" height="12" viewBox="0 0 24 24" stroke="var(--text-muted)"
			><path d="M12 2 L14 10 L22 12 L14 14 L12 22 L10 14 L2 12 L10 10 Z" /></svg
		>
	</span>
	<span class="sparkle" aria-hidden="true" style="bottom:28%;left:16%">
		<svg width="10" height="10" viewBox="0 0 24 24" stroke="var(--border-light)"
			><path d="M12 2 L14 10 L22 12 L14 14 L12 22 L10 14 L2 12 L10 10 Z" /></svg
		>
	</span>
	<span class="sparkle" aria-hidden="true" style="top:30%;left:10%">
		<svg width="8" height="8" viewBox="0 0 24 24" stroke="var(--accent)"
			><path d="M12 2 L14 10 L22 12 L14 14 L12 22 L10 14 L2 12 L10 10 Z" /></svg
		>
	</span>
	<span class="sparkle" aria-hidden="true" style="top:72%;right:8%">
		<svg width="11" height="11" viewBox="0 0 24 24" stroke="var(--text-muted)"
			><path d="M12 2 L14 10 L22 12 L14 14 L12 22 L10 14 L2 12 L10 10 Z" /></svg
		>
	</span>
	<span class="sparkle" aria-hidden="true" style="top:12%;left:22%">
		<svg width="7" height="7" viewBox="0 0 24 24" stroke="var(--border-light)"
			><path d="M12 2 L14 10 L22 12 L14 14 L12 22 L10 14 L2 12 L10 10 Z" /></svg
		>
	</span>
	<span class="sparkle" aria-hidden="true" style="bottom:18%;right:28%">
		<svg width="14" height="14" viewBox="0 0 24 24" stroke="var(--accent)"
			><path d="M12 2 L14 10 L22 12 L14 14 L12 22 L10 14 L2 12 L10 10 Z" /></svg
		>
	</span>
	<span class="sparkle" aria-hidden="true" style="top:44%;left:5%">
		<svg width="9" height="9" viewBox="0 0 24 24" stroke="var(--text-muted)"
			><path d="M12 2 L14 10 L22 12 L14 14 L12 22 L10 14 L2 12 L10 10 Z" /></svg
		>
	</span>

	<Aperture />

	<div class="hero-center">
		<h1 class="hero-name">Ming Chong Lim<span class="dot">.</span></h1>
		<div class="hero-subtitle">Quantitative Developer &middot; DRW</div>
	</div>

	<div class="hero-meta-row">
		<span>Singapore</span><span>&middot;</span>
		<span><a href="https://drw.com/" target="_blank" rel="noopener noreferrer">DRW</a></span><span>&middot;</span>
		<span><a href="/Ming Chong Lim_Resume.pdf" target="_blank" rel="noopener noreferrer" class="resume-link">
			<svg class="resume-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12l7 7 7-7"/></svg>
			R&eacute;sum&eacute;
		</a></span>
	</div>

	<div class="hero-scroll">
		<span>Scroll</span>
		<div class="scroll-tick"></div>
	</div>

	<div class="social-rail">
		<a href="https://github.com/qwoprocks" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
			<svg viewBox="0 0 24 24"
				><path
					d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"
				/></svg
			>
		</a>
		<a href="https://www.linkedin.com/in/ming-chong-lim/" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
			<svg viewBox="0 0 24 24"
				><path
					d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"
				/></svg
			>
		</a>
		<a href="mailto:limmingchong98@gmail.com" aria-label="Email">
			<svg viewBox="0 0 24 24"
				><path
					d="M1.5 8.67v8.58a3 3 0 0 0 3 3h15a3 3 0 0 0 3-3V8.67l-8.928 5.493a3 3 0 0 1-3.144 0L1.5 8.67Z"
				/><path
					d="M22.5 6.908V6.75a3 3 0 0 0-3-3h-15a3 3 0 0 0-3 3v.158l9.714 5.978a1.5 1.5 0 0 0 1.572 0L22.5 6.908Z"
				/></svg
			>
		</a>
	</div>
</section>

<style>
	.hero {
		min-height: 100vh;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		position: relative;
		overflow: hidden;
		transition: var(--theme-transition);
	}

	.sparkle {
		position: absolute;
		pointer-events: none;
		z-index: 1;
		transition: opacity 0.5s;
	}

	:global([data-theme='light']) .sparkle {
		opacity: 0;
	}

	.sparkle :global(svg) {
		fill: none;
		stroke-width: 1;
		animation: sparkleFloat 5s ease-in-out infinite;
	}

	.sparkle:nth-child(3) :global(svg) {
		animation-delay: 1.5s;
	}

	.sparkle:nth-child(4) :global(svg) {
		animation-delay: 3s;
	}

	.sparkle:nth-child(5) :global(svg) {
		animation-delay: 4.2s;
	}

	.sparkle:nth-child(6) :global(svg) {
		animation-delay: 0.8s;
	}

	.sparkle:nth-child(7) :global(svg) {
		animation-delay: 2.5s;
	}

	.sparkle:nth-child(8) :global(svg) {
		animation-delay: 3.8s;
	}

	.sparkle:nth-child(9) :global(svg) {
		animation-delay: 1.2s;
	}

	@keyframes sparkleFloat {
		0%,
		100% {
			transform: scale(1) rotate(0deg);
			opacity: 0.15;
		}
		50% {
			transform: scale(1.3) rotate(20deg);
			opacity: 0.45;
		}
	}

	.hero-center {
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		text-align: center;
		z-index: 5;
		pointer-events: none;
	}

	.hero-name {
		font-family: var(--font-head);
		font-weight: 300;
		font-size: clamp(2.5rem, 5vw, 3.5rem);
		color: var(--heading);
		line-height: 1.15;
		letter-spacing: -0.03em;
		white-space: nowrap;
		transition: color 0.5s;
	}

	.hero-name .dot {
		color: var(--accent);
		transition: color 0.5s;
	}

	.hero-subtitle {
		font-size: 0.65rem;
		font-weight: 400;
		color: var(--text-muted);
		letter-spacing: 0.15em;
		text-transform: uppercase;
		margin-top: 0.8rem;
		transition: color 0.5s;
	}

	.hero-meta-row {
		position: absolute;
		bottom: 25%;
		left: 50%;
		transform: translateX(-50%);
		display: flex;
		gap: 2rem;
		z-index: 5;
	}

	.hero-meta-row span {
		font-size: 0.68rem;
		color: var(--text-muted);
		letter-spacing: 0.05em;
		transition: color 0.5s;
	}

	.hero-meta-row a {
		color: var(--text-dim);
		text-decoration: none;
		transition: color 0.3s;
		position: relative;
		padding-bottom: 2px;
	}

	.hero-meta-row a::after {
		content: '';
		position: absolute;
		bottom: 0;
		left: 0;
		width: 0;
		height: 1px;
		background: var(--accent);
		transition: width 0.4s var(--ease);
	}

	.hero-meta-row a:hover {
		color: var(--accent);
	}

	.hero-meta-row a:hover::after {
		width: 100%;
	}

	.resume-link {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
	}

	.resume-icon {
		width: 12px;
		height: 12px;
		stroke: currentColor;
		transition: transform 0.3s var(--ease);
	}

	.resume-link:hover .resume-icon {
		transform: translateY(2px);
	}

	.hero-scroll {
		position: absolute;
		bottom: 2rem;
		left: 50%;
		transform: translateX(-50%);
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.5rem;
	}

	.hero-scroll span {
		font-size: 0.5rem;
		letter-spacing: 0.2em;
		text-transform: uppercase;
		color: var(--text-muted);
	}

	.scroll-tick {
		width: 1px;
		height: 28px;
		background: linear-gradient(to bottom, var(--border-light), transparent);
		position: relative;
		overflow: hidden;
	}

	.scroll-tick::after {
		content: '';
		position: absolute;
		top: -100%;
		width: 1px;
		height: 100%;
		background: var(--accent);
		animation: scrollPulse 2s ease infinite;
	}

	@keyframes scrollPulse {
		0%,
		100% {
			top: -100%;
		}
		50% {
			top: 100%;
		}
	}

	.social-rail {
		position: absolute;
		right: 2.5rem;
		top: 50%;
		transform: translateY(-50%);
		display: flex;
		flex-direction: column;
		gap: 1rem;
		border-left: 1px solid var(--border);
		padding-left: 1rem;
		z-index: 3;
	}

	.social-rail a {
		color: var(--text-muted);
		transition: color 0.3s;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.social-rail a:hover {
		color: var(--accent);
	}

	.social-rail :global(svg) {
		width: 15px;
		height: 15px;
		fill: currentColor;
	}

	@media (max-width: 768px) {
		.social-rail {
			display: none;
		}
	}

	@media (max-width: 480px) {
		.hero-meta-row {
			display: none;
		}
	}
</style>
