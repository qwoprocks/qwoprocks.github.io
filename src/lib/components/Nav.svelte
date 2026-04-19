<script lang="ts">
	import { onMount } from 'svelte';
	import { afterNavigate } from '$app/navigation';
	import { page } from '$app/state';
	import ThemeToggle from './ThemeToggle.svelte';

	let scrolled = $state(false);
	let menuOpen = $state(false);
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
		menuOpen = false;
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
		<button
			class="hamburger"
			class:open={menuOpen}
			onclick={() => (menuOpen = !menuOpen)}
			aria-label="Toggle navigation menu"
			aria-expanded={menuOpen}
		>
			<span></span>
			<span></span>
			<span></span>
		</button>
	</div>
</nav>

{#if menuOpen}
	<div class="mobile-overlay" onclick={() => (menuOpen = false)} role="presentation"></div>
	<div class="mobile-menu">
		<ul>
			<li><a href="/#about" onclick={() => (menuOpen = false)}>About</a></li>
			<li><a href="/#experience" onclick={() => (menuOpen = false)}>Experience</a></li>
			<li><a href="/#education" onclick={() => (menuOpen = false)}>Education</a></li>
			<li><a href="/#awards" onclick={() => (menuOpen = false)}>Awards</a></li>
			<li><a href="/#projects" onclick={() => (menuOpen = false)}>Projects</a></li>
			<li><a href="/#skills" onclick={() => (menuOpen = false)}>Skills</a></li>
			<li><a href="/gallery" data-sveltekit-reload onclick={() => (menuOpen = false)}>Gallery</a></li>
		</ul>
	</div>
{/if}

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

	/* Hamburger button — mobile only */
	.hamburger {
		display: none;
		flex-direction: column;
		justify-content: center;
		gap: 4px;
		width: 30px;
		height: 30px;
		background: transparent;
		border: 1px solid var(--border);
		border-radius: 50%;
		cursor: pointer;
		padding: 7px;
		transition: var(--theme-transition);
		flex-shrink: 0;
	}

	.hamburger:hover {
		border-color: var(--accent);
	}

	.hamburger span {
		display: block;
		width: 100%;
		height: 1px;
		background: var(--text-dim);
		transition: transform 0.3s, opacity 0.3s;
	}

	.hamburger:hover span {
		background: var(--accent);
	}

	.hamburger.open span:nth-child(1) {
		transform: translateY(5px) rotate(45deg);
	}

	.hamburger.open span:nth-child(2) {
		opacity: 0;
	}

	.hamburger.open span:nth-child(3) {
		transform: translateY(-5px) rotate(-45deg);
	}

	@media (max-width: 480px) {
		.nav-links {
			display: none;
		}

		.hamburger {
			display: flex;
		}
	}

	.mobile-overlay {
		position: fixed;
		inset: 0;
		z-index: 98;
		background: rgba(0, 0, 0, 0.3);
	}

	.mobile-menu {
		position: fixed;
		top: 56px;
		right: 0;
		z-index: 99;
		background: var(--surface);
		border: 1px solid var(--border);
		border-top: none;
		border-radius: 0 0 0 8px;
		padding: 1rem 2rem;
		transition: var(--theme-transition);
	}

	.mobile-menu ul {
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.mobile-menu a {
		font-size: 0.7rem;
		font-weight: 400;
		letter-spacing: 0.06em;
		color: var(--text-muted);
		text-decoration: none;
		transition: color 0.3s;
	}

	.mobile-menu a:hover {
		color: var(--accent);
	}
</style>
