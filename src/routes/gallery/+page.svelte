<script lang="ts">
	import Footer from '$lib/components/Footer.svelte';
	import { galleryItems } from '$lib/data/gallery';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { onMount } from 'svelte';

	const tabParam = $derived(page.url.searchParams.get('tab'));
	let activeId = $derived(
		tabParam && galleryItems.some((i) => i.id === tabParam) ? tabParam : galleryItems[0].id
	);
	const activeItem = $derived(galleryItems.find((i) => i.id === activeId)!);

	let expandedSections = $state<Record<string, boolean>>({});

	function toggleSection(title: string) {
		expandedSections = { ...expandedSections, [title]: !expandedSections[title] };
	}

	// --- Grass scene filter state ---
	interface FilterEntry {
		pass: { enabled: boolean };
		enabled: boolean;
		name: string;
	}

	let filtersReady = $state(false);
	let filterKeys = $state<string[]>([]);
	let filterStates = $state<Record<string, boolean>>({});

	function pollForFilters() {
		const w = window as any;
		if (w.__galleryFilters && w.__galleryToggleFilter) {
			const filters: Record<string, FilterEntry> = w.__galleryFilters;
			filterKeys = Object.keys(filters);
			const states: Record<string, boolean> = {};
			for (const key of filterKeys) {
				states[key] = filters[key].enabled;
			}
			filterStates = states;
			filtersReady = true;
		} else {
			setTimeout(pollForFilters, 200);
		}
	}

	function toggleFilter(key: string) {
		const newState = !filterStates[key];
		const w = window as any;
		if (newState) {
			for (const k of filterKeys) {
				if (k !== key && filterStates[k]) {
					filterStates[k] = false;
					w.__galleryToggleFilter(k, false);
				}
			}
		}
		filterStates = { ...filterStates, [key]: newState };
		w.__galleryToggleFilter(key, newState);
	}

	function getFilterName(key: string): string {
		const w = window as any;
		if (w.__galleryFilters && w.__galleryFilters[key]) {
			return w.__galleryFilters[key].name;
		}
		return key;
	}

	// --- Maze script loading ---
	let mazeLoaded = $state(false);

	function loadMaze() {
		if (mazeLoaded) return;
		const script = document.createElement('script');
		script.src = '/js/maze/maze.js';
		script.onload = () => {
			mazeLoaded = true;
		};
		document.body.appendChild(script);
	}

	// --- Tab switching ---
	function selectItem(id: string) {
		expandedSections = {};
		goto(`?tab=${id}`, { noScroll: true, keepFocus: true });
	}

	onMount(() => {
		pollForFilters();
		// Load maze eagerly — 11KB, loads in background while user views grass scene
		loadMaze();
	});
</script>

<svelte:head>
	<title>Gallery | Ming Chong Lim</title>
	<script type="module" src="/js/grass/main.js"></script>
	<style>
		.wrapper section canvas {
			position: relative;
			z-index: 1;
		}
		/* Ensure source code button sits above the canvas */
		.sourceCodeButton {
			position: relative;
			z-index: 2;
		}
	</style>
</svelte:head>

<div class="gallery-layout">
	<!-- Sidebar / Tab bar -->
	<nav class="gallery-sidebar" aria-label="Gallery experiments">
		{#each galleryItems as item}
			<button
				class="sidebar-item"
				class:active={activeId === item.id}
				onclick={() => selectItem(item.id)}
			>
				<span class="sidebar-title">{item.title}</span>
				<span class="sidebar-date">{item.date}</span>
			</button>
		{/each}
	</nav>

	<!-- Main content -->
	<div class="gallery-main">
		<!-- Grass scene -->
		<div class="experiment-panel" class:hidden={activeId !== 'grass-scene'}>
			<div class="wrapper">
				<section>
					<div class="canvas-placeholder">
						<div class="placeholder-pulse"></div>
						<span class="placeholder-label">Loading scene...</span>
					</div>
				</section>
			</div>

			{#if filtersReady && activeId === 'grass-scene'}
				<div class="wrapper">
					<div class="filter-bar">
						Filters:
						{#each filterKeys as key}
							<button
								class="filter-btn"
								class:active={filterStates[key]}
								onclick={() => toggleFilter(key)}
							>
								{getFilterName(key)}
							</button>
						{/each}
					</div>
				</div>
			{/if}
		</div>

		<!-- Maze generator -->
		<div class="experiment-panel" class:hidden={activeId !== 'maze-generator'}>
			<div class="wrapper">
				<div class="maze-section">
					<div class="maze-container"></div>
				</div>
			</div>
		</div>

		<!-- Writeup for active item -->
		<div class="container writeup-container">
			<div class="writeup-header">
				<h1>{activeItem.title}</h1>
				<div class="writeup-meta">
					<span class="writeup-date">{activeItem.date}</span>
					<div class="writeup-tags">
						{#each activeItem.tags as tag}
							<span class="tag">{tag}</span>
						{/each}
					</div>
				</div>
				<p class="writeup-desc">{activeItem.description}</p>
			</div>

			{#each activeItem.sections as section}
				<div class="writeup-section">
					<button
						class="section-toggle"
						onclick={() => toggleSection(section.title)}
						aria-expanded={expandedSections[section.title] ?? false}
					>
						<h2>{section.title}</h2>
						<span class="toggle-icon" class:expanded={expandedSections[section.title]}>+</span>
					</button>
					<div class="section-body" class:expanded={expandedSections[section.title]}>
						<div class="section-body-inner">
							{@html section.content}
						</div>
					</div>
				</div>
			{/each}

			{#if activeItem.references.length > 0}
				<div class="writeup-section">
					<button
						class="section-toggle"
						onclick={() => toggleSection('references')}
						aria-expanded={expandedSections['references'] ?? false}
					>
						<h2>References</h2>
						<span class="toggle-icon" class:expanded={expandedSections['references']}>+</span>
					</button>
					<div class="section-body" class:expanded={expandedSections['references']}>
						<div class="section-body-inner">
							<ul class="references-list">
								{#each activeItem.references as ref}
									<li><a href={ref.url} target="_blank" rel="noopener noreferrer">{ref.label}</a></li>
								{/each}
							</ul>
						</div>
					</div>
				</div>
			{/if}

			<Footer />
		</div>
	</div>
</div>

<style>
	/* ===== Gallery layout ===== */
	.gallery-layout {
		display: flex;
		min-height: 100vh;
	}

	/* ===== Sidebar ===== */
	.gallery-sidebar {
		width: 180px;
		flex-shrink: 0;
		padding: 80px 0 2rem;
		border-right: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		gap: 0;
		position: sticky;
		top: 0;
		height: 100vh;
		overflow-y: auto;
		transition: var(--theme-transition);
	}

	.sidebar-item {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		padding: 0.8rem 1rem;
		border: none;
		border-left: 2px solid transparent;
		background: none;
		cursor: pointer;
		text-align: left;
		transition: border-color 0.3s, background 0.3s, color 0.3s;
	}

	.sidebar-item:hover {
		background: var(--accent-dim);
	}

	.sidebar-item.active {
		border-left-color: var(--accent);
		background: var(--accent-dim);
	}

	.sidebar-title {
		font-family: var(--font-mono);
		font-size: var(--text-base);
		letter-spacing: 0.04em;
		color: var(--text-muted);
		transition: color 0.3s;
	}

	.sidebar-item.active .sidebar-title {
		color: var(--accent);
	}

	.sidebar-item:hover .sidebar-title {
		color: var(--text);
	}

	.sidebar-date {
		font-family: var(--font-mono);
		font-size: var(--text-sm);
		letter-spacing: 0.06em;
		color: var(--text-muted);
		opacity: 0.6;
	}

	/* ===== Main content ===== */
	.gallery-main {
		flex: 1;
		min-width: 0;
	}

	.experiment-panel {
		transition: opacity 0.3s;
	}

	.experiment-panel.hidden {
		display: none;
	}

	.wrapper {
		max-width: 960px;
		margin: 0 auto;
		padding: 0 3rem;
	}

	section {
		width: 100%;
		margin-top: 80px;
		position: relative;
		aspect-ratio: 4 / 3;
	}

	/* Maze section — no aspect ratio, just hosts the container */
	.maze-section {
		margin-top: 80px;
		display: flex;
		justify-content: center;
		position: relative;
		min-height: 300px;
	}

	/* Placeholder while canvas loads */
	.canvas-placeholder {
		position: absolute;
		inset: 0;
		z-index: 0;
		background: var(--surface);
		border: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 1rem;
		transition: var(--theme-transition);
	}

	.placeholder-pulse {
		width: 32px;
		height: 32px;
		border: 1.5px solid var(--border-light);
		border-top-color: var(--accent);
		border-radius: 50%;
		animation: spin 1s linear infinite;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	.placeholder-label {
		font-size: var(--text-xs);
		letter-spacing: 0.1em;
		text-transform: uppercase;
		color: var(--text-muted);
	}

	/* Filter toggle bar */
	.filter-bar {
		display: flex;
		flex-wrap: wrap;
		font-family: var(--font-mono);
		font-size: var(--text-lg);
		gap: 0.5rem;
		padding: 0.75rem 0;
		justify-content: flex-start;
	}

	.filter-btn {
		font-family: var(--font-mono);
		font-size: var(--text-xs);
		letter-spacing: 0.06em;
		padding: 0.3rem 0.7rem;
		border: 1px solid var(--border);
		border-radius: 999px;
		background: transparent;
		color: var(--text-muted);
		cursor: pointer;
		transition: border-color 0.3s, color 0.3s, background 0.3s;
		user-select: none;
		white-space: nowrap;
	}

	.filter-btn:hover {
		border-color: var(--border-light);
		color: var(--text);
	}

	.filter-btn.active {
		border-color: var(--accent);
		color: var(--accent);
		background: var(--accent-dim);
	}

	/* Canvas injected by main.js */
	section :global(canvas) {
		width: 100% !important;
		height: auto !important;
		display: block;
	}

	:global(.sourceCodeButton) {
		float: right;
		background: var(--surface);
		border: 1px solid var(--border);
		opacity: 0.9;
		border-radius: 50%;
		margin-top: -50px;
		margin-right: 10px;
		padding: 10px;
		box-shadow: none;
		-webkit-transform: translateZ(0);
		transition: var(--theme-transition);
		position: relative;
		z-index: 2;
	}

	:global(.sourceCodeButton:hover) {
		opacity: 1;
		border-color: var(--accent);
	}

	:global(.sourceCodeButton img) {
		display: block;
		width: 15px;
		height: 15px;
		filter: var(--icon-invert, none);
	}

	:global([data-theme='dark'] .sourceCodeButton img) {
		filter: invert(1) brightness(0.8);
	}

	/* Writeup */
	.writeup-container {
		padding-top: 3rem;
	}

	.writeup-header {
		margin-bottom: 2rem;
	}

	.writeup-header h1 {
		font-family: var(--font-head);
		font-weight: 400;
		font-size: var(--text-4xl);
		color: var(--heading);
		margin-bottom: 0.5rem;
		transition: color 0.5s;
	}

	.writeup-meta {
		display: flex;
		align-items: center;
		gap: 1rem;
		margin-bottom: 0.8rem;
		flex-wrap: wrap;
	}

	.writeup-date {
		font-size: var(--text-base);
		color: var(--text-muted);
		letter-spacing: 0.02em;
	}

	.writeup-tags {
		display: flex;
		gap: 0.4rem;
		flex-wrap: wrap;
	}

	.tag {
		font-size: var(--text-2xs);
		padding: 0.1rem 0.4rem;
		border: 1px solid var(--border);
		color: var(--text-muted);
		letter-spacing: 0.05em;
		transition: var(--theme-transition);
	}

	.writeup-desc {
		font-size: var(--text-xl);
		color: var(--text-dim);
		line-height: 1.7;
		transition: color 0.5s;
	}

	/* Sections */
	.writeup-section {
		border-top: 1px solid var(--border);
	}

	.section-toggle {
		display: flex;
		justify-content: space-between;
		align-items: center;
		width: 100%;
		padding: 0.8rem 0.6rem;
		background: none;
		border: none;
		border-radius: 4px;
		cursor: pointer;
		text-align: left;
		transition: background 0.3s;
	}

	.section-toggle:hover {
		background: var(--accent-dim);
	}

	.section-toggle h2 {
		font-family: var(--font-head);
		font-weight: 400;
		font-size: var(--text-2xl);
		color: var(--heading);
		margin: 0;
		pointer-events: none;
		transition: color 0.5s;
	}

	.toggle-icon {
		font-size: var(--text-2xl);
		color: var(--text-muted);
		transition: transform 0.3s var(--ease), color 0.3s;
		font-family: var(--font-mono);
		user-select: none;
		flex-shrink: 0;
		line-height: 1;
	}

	.toggle-icon.expanded {
		transform: rotate(45deg);
	}

	.section-toggle:hover .toggle-icon {
		color: var(--accent);
	}

	/* Expandable body */
	.section-body {
		display: grid;
		grid-template-rows: 0fr;
		transition: grid-template-rows 0.4s var(--ease);
	}

	.section-body.expanded {
		grid-template-rows: 1fr;
	}

	.section-body-inner {
		overflow: hidden;
		min-height: 0;
	}

	.section-body.expanded .section-body-inner {
		padding: 0.6rem 0.6rem 1.5rem;
	}

	/* Content styling */
	.section-body-inner :global(p) {
		font-size: var(--text-lg);
		color: var(--text);
		line-height: 1.8;
		margin: 0 0 0.8rem;
		transition: color 0.5s;
	}

	.section-body-inner :global(p:last-child) {
		margin-bottom: 0;
	}

	.section-body-inner :global(code) {
		font-size: var(--text-lg);
		padding: 0.1rem 0.35rem;
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: 2px;
		font-family: var(--font-mono);
		transition: var(--theme-transition);
	}

	.section-body-inner :global(ol) {
		padding-left: 1.5rem;
		margin: 0.5rem 0 0.8rem;
	}

	.section-body-inner :global(li) {
		font-size: var(--text-lg);
		color: var(--text);
		line-height: 1.7;
		margin-bottom: 0.3rem;
		transition: color 0.5s;
	}

	.section-body-inner :global(em) {
		font-style: italic;
	}

	/* References */
	.references-list {
		list-style: none;
		padding: 0;
		margin: 0;
	}

	.references-list li {
		padding: 0.4rem 0;
		border-bottom: 1px solid var(--border);
		font-size: var(--text-lg);
	}

	.references-list li:last-child {
		border-bottom: none;
	}

	.references-list a {
		color: var(--accent);
		text-decoration: none;
		border-bottom: 1px solid var(--accent-soft);
		transition: border-color 0.3s, color 0.5s;
	}

	.references-list a:hover {
		border-color: var(--accent);
	}

	/* ===== Mobile: collapse sidebar to horizontal tabs ===== */
	@media (max-width: 768px) {
		.gallery-layout {
			flex-direction: column;
		}

		.gallery-sidebar {
			width: 100%;
			height: auto;
			position: relative;
			flex-direction: row;
			padding: 56px 0 0;
			border-right: none;
			border-bottom: 1px solid var(--border);
			overflow-x: auto;
			overflow-y: hidden;
			gap: 0;
		}

		.sidebar-item {
			border-left: none;
			border-bottom: 2px solid transparent;
			padding: 0.6rem 1rem;
			white-space: nowrap;
			flex-shrink: 0;
		}

		.sidebar-item.active {
			border-left-color: transparent;
			border-bottom-color: var(--accent);
		}

		.wrapper {
			padding: 0 1.5rem;
		}
	}
</style>
