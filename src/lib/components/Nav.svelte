<script lang="ts">
	import { onMount } from 'svelte';
	import { afterNavigate } from '$app/navigation';
	import { page } from '$app/state';
	import ThemeToggle from './ThemeToggle.svelte';

	let scrolled = $state(false);
	let observer: IntersectionObserver | null = null;

	const isHome = $derived(page.url.pathname === '/');
	const isDetailPage = $derived(
		!isHome &&
		['/experience', '/education', '/awards', '/projects', '/skills', '/gallery'].includes(
			page.url.pathname
		)
	);

	function setupHeroObserver() {
		observer?.disconnect();
		const hero = document.querySelector('.hero');
		if (!hero) {
			scrolled = true;
			return;
		}
		scrolled = false;
		observer = new IntersectionObserver(
			([e]) => {
				scrolled = !e.isIntersecting;
			},
			{ threshold: 0.1 }
		);
		observer.observe(hero);
	}

	onMount(() => {
		setupHeroObserver();
		return () => observer?.disconnect();
	});

	afterNavigate(() => {
		setupHeroObserver();
	});
</script>

<nav class="nav" class:scrolled>
	{#if isDetailPage}
		<a href="/" class="nav-back" aria-label="Back to home">
			<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
				<line x1="19" y1="12" x2="5" y2="12" />
				<polyline points="12 19 5 12 12 5" />
			</svg>
		</a>
	{:else}
		<a href="/" class="nav-logo">MCL</a>
	{/if}
	<div class="nav-right">
		<ul class="nav-links">
			<li><a href="/#about">About</a></li>
			<li><a href="/#experience">Experience</a></li>
			<li><a href="/#education">Education</a></li>
			<li><a href="/#awards">Awards</a></li>
			<li><a href="/#projects">Projects</a></li>
			<li><a href="/#skills">Skills</a></li>
			<li><a href="/gallery" data-sveltekit-reload>Gallery</a></li>
		</ul>
		<ThemeToggle />
	</div>
</nav>

<style>
	.nav {
		position: fixed;
		top: 0;
		left: 0;
		right: 0;
		z-index: 100;
		padding: 0 3rem;
		height: 56px;
		display: flex;
		align-items: center;
		justify-content: space-between;
		border-bottom: 1px solid transparent;
		transition: var(--theme-transition), backdrop-filter 0.5s ease;
		background: transparent;
	}

	.nav.scrolled {
		background: color-mix(in srgb, var(--bg) 92%, transparent);
		backdrop-filter: blur(16px);
		-webkit-backdrop-filter: blur(16px);
		border-color: var(--border);
	}

	/* Logo — hidden until scrolled past hero */
	.nav-logo {
		font-family: var(--font-head);
		font-weight: 500;
		font-size: 0.9rem;
		color: var(--heading);
		text-decoration: none;
		letter-spacing: 0.02em;
		opacity: 0;
		pointer-events: none;
		transition: opacity 0.5s, color 0.5s;
	}

	.nav.scrolled .nav-logo {
		opacity: 1;
		pointer-events: auto;
	}

	/* Back arrow on detail pages */
	.nav-back {
		display: flex;
		align-items: center;
		color: var(--text-dim);
		text-decoration: none;
		transition: color 0.3s;
	}

	.nav-back:hover {
		color: var(--accent);
	}

	.nav-back svg {
		width: 18px;
		height: 18px;
	}

	.nav-right {
		display: flex;
		align-items: center;
		gap: 2rem;
	}

	.nav-links {
		display: flex;
		gap: 2.5rem;
		list-style: none;
	}

	.nav-links a {
		font-size: 0.6rem;
		font-weight: 400;
		letter-spacing: 0.06em;
		color: var(--text-muted);
		text-decoration: none;
		transition: color 0.3s;
		position: relative;
		padding-bottom: 3px;
	}

	.nav-links a::after {
		content: '';
		position: absolute;
		bottom: 0;
		left: 0;
		width: 0;
		height: 1px;
		background: var(--accent);
		transition: width 0.4s var(--ease);
	}

	.nav-links a:hover {
		color: var(--text);
	}

	.nav-links a:hover::after {
		width: 100%;
	}

	@media (max-width: 768px) {
		.nav {
			padding: 0 1.5rem;
		}

		.nav-links {
			gap: 1.2rem;
		}

		.nav-links a {
			font-size: 0.5rem;
		}
	}

	@media (max-width: 480px) {
		.nav-links {
			display: none;
		}
	}
</style>
