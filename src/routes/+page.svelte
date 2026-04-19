<script lang="ts">
	import Hero from '$lib/components/Hero.svelte';
	import SectionHeader from '$lib/components/SectionHeader.svelte';
	import ShutterLine from '$lib/components/ShutterLine.svelte';
	import TimelineItem from '$lib/components/TimelineItem.svelte';
	import ProjectCard from '$lib/components/ProjectCard.svelte';
	import SkillBar from '$lib/components/SkillBar.svelte';
	import Footer from '$lib/components/Footer.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { experiences } from '$lib/data/experiences';
	import { education } from '$lib/data/education';
	import { awards } from '$lib/data/awards';
	import { projects } from '$lib/data/projects';
	import { skillGroups } from '$lib/data/skills';

	const topExperiences = experiences.slice(0, 3);
	const topEducation = education.slice(0, 2);
	const topAwards = awards.slice(0, 3);
	const topProjects = projects.slice(0, 3);

	let eduExpanded: Record<number, boolean> = $state({});
	function toggleEdu(i: number) {
		eduExpanded[i] = !eduExpanded[i];
	}
</script>

<svelte:head>
	<title>Home | Ming Chong Lim</title>
</svelte:head>

<Hero />

<div class="container">
	<!-- About -->
	<section class="section" id="about">
		<ShutterLine />
		<SectionHeader num="01" title="About" />
		<div class="about-text reveal reveal-d1" use:reveal>
			<p>
				I'm a Quantitative Developer at <a href="https://drw.com/" target="_blank" rel="noopener noreferrer">DRW</a> in
				Singapore. I graduated from
				<a
					href="https://www.ml.cmu.edu/academics/primary-ms-machine-learning-masters.html"
					target="_blank" rel="noopener noreferrer">Carnegie Mellon University</a
				> with a Master's in Machine Learning.
			</p>
			<p>
				While ML is my main focus, I have a broad range of interests spanning across multiple
				computing disciplines, and in general, I enjoy solving challenging problems and devising
				elegant solutions to them.
			</p>
			<p>
				I'm passionate about both learning and teaching. I served as a teaching assistant for various
				modules over 6 semesters at both <a href="/experience#teaching-assistant-1">NUS</a> and
				<a href="/experience#teaching-assistant-2">CMU</a>.
			</p>
		</div>
	</section>

	<!-- Experience -->
	<section class="section" id="experience">
		<ShutterLine />
		<SectionHeader num="02" title="Experience" />
		<div class="timeline reveal reveal-d1" use:reveal>
			{#each topExperiences as exp}
				<TimelineItem experience={exp} expandable />
			{/each}
		</div>
		<div class="view-all reveal reveal-d2" use:reveal>
			<a href="/experience">View all &rarr;</a>
		</div>
	</section>

	<!-- Education -->
	<section class="section" id="education">
		<ShutterLine />
		<SectionHeader num="03" title="Education" />
		{#each topEducation as edu, i}
			<div
				class="edu-row expandable reveal reveal-d{i + 1}"
				class:expanded={eduExpanded[i]}
				use:reveal
				onclick={() => toggleEdu(i)}
				onkeydown={(e) => e.key === 'Enter' && toggleEdu(i)}
				role="button"
				tabindex={0}
			>
				<div class="edu-year">{edu.period}</div>
				<div class="edu-divider"></div>
				<div class="edu-info">
					<h3>{edu.school}</h3>
					<div class="degree">{edu.degree}</div>
					<div class="note">{edu.note}</div>
					<div class="edu-detail-hint">
						<svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
							<polyline points="6 9 12 15 18 9" />
						</svg>
					</div>
					<div class="edu-expand">
						<div class="edu-expand-inner">
							{#if edu.gpa}
								<div class="edu-gpa">GPA: {edu.gpa}</div>
							{/if}
							<ul class="edu-details">
								{#each edu.details as detail}
									<li>{@html detail}</li>
								{/each}
							</ul>
						</div>
					</div>
				</div>
			</div>
		{/each}
		<div class="view-all reveal reveal-d3" use:reveal>
			<a href="/education">View all &rarr;</a>
		</div>
	</section>

	<!-- Awards -->
	<section class="section" id="awards">
		<ShutterLine />
		<SectionHeader num="04" title="Awards" />
		{#each topAwards as award, i}
			<div class="award-row reveal reveal-d{i + 1}" use:reveal>
				<div class="award-place">{award.place}</div>
				<div class="award-info">
					<h3>{award.title}</h3>
					<p>{award.description}</p>
				</div>
			</div>
		{/each}
		<div class="view-all reveal" use:reveal>
			<a href="/awards">View all &rarr;</a>
		</div>
	</section>

	<!-- Projects -->
	<section class="section" id="projects">
		<ShutterLine />
		<SectionHeader num="05" title="Projects" />
		<div class="projects-grid reveal reveal-d1" use:reveal>
			{#each topProjects as project}
				<ProjectCard {project} />
			{/each}
		</div>
		<div class="view-all reveal" use:reveal>
			<a href="/projects">View all &rarr;</a>
		</div>
	</section>

	<!-- Skills -->
	<section class="section" id="skills">
		<ShutterLine />
		<SectionHeader num="06" title="Skills" />
		<div class="skills-spec reveal reveal-d1" use:reveal>
			{#each skillGroups as group}
				<div class="spec-group">
					<h3>{group.title}</h3>
					{#each group.skills as skill}
						<SkillBar {skill} />
					{/each}
				</div>
			{/each}
		</div>
	</section>

	<!-- Gallery CTA -->
	<section class="section" id="gallery">
		<ShutterLine />
		<div class="gallery-band reveal" use:reveal>
			<div>
				<h3>Step into the Gallery</h3>
				<p>A collection of experiments and personal projects — procedural art, interactive demos, and whatever else I'm tinkering with.</p>
			</div>
			<a href="/gallery" class="gallery-link" data-sveltekit-reload>
				Explore
				<svg viewBox="0 0 24 24"
					><line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" /></svg
				>
			</a>
		</div>
	</section>

	<Footer />
</div>

<style>
	/* About */
	.about-text {
		max-width: 560px;
		font-size: var(--text-xl);
		line-height: 1.85;
		font-weight: 300;
		color: var(--text);
		transition: color 0.5s;
	}

	.about-text :global(p + p) {
		margin-top: 1rem;
	}

	.about-text :global(a) {
		color: var(--accent);
		text-decoration: none;
		border-bottom: 1px solid var(--accent-soft);
		transition: border-color 0.3s, color 0.5s;
	}

	.about-text :global(a:hover) {
		border-color: var(--accent);
	}

	/* Timeline */
	.timeline {
		position: relative;
		padding-left: 2rem;
	}

	.timeline::before {
		content: '';
		position: absolute;
		left: 4px;
		top: 8px;
		bottom: 8px;
		width: 1px;
		background: linear-gradient(to bottom, var(--border-light), var(--border), transparent);
		transition: background 0.5s;
	}

	/* Education */
	.edu-row {
		display: grid;
		grid-template-columns: 100px 1px 1fr;
		gap: 1.5rem;
		padding: 1.5rem 0;
		align-items: start;
		transition: var(--theme-transition);
	}

	.edu-row + .edu-row {
		border-top: 1px solid var(--border);
	}

	.edu-year {
		font-size: var(--text-base);
		color: var(--text-muted);
		text-align: right;
		padding-top: 0.15rem;
	}

	.edu-divider {
		width: 1px;
		height: 100%;
		background: var(--border-light);
		min-height: 40px;
		transition: background 0.5s;
	}

	.edu-info h3 {
		font-family: var(--font-head);
		font-weight: 500;
		font-size: var(--text-2xl);
		color: var(--heading);
		margin-bottom: 0.1rem;
		transition: color 0.5s;
	}

	.degree {
		font-size: var(--text-lg);
		color: var(--accent);
		font-weight: 400;
		margin-bottom: 0.2rem;
		font-family: var(--font-head);
		transition: color 0.5s;
	}

	.note {
		font-size: var(--text-md);
		color: var(--text-dim);
		transition: color 0.5s;
	}

	/* Expandable education rows */
	.edu-row.expandable {
		cursor: pointer;
		transition: var(--theme-transition), background 0.3s;
	}

	.edu-row.expandable:hover {
		background: var(--accent-dim);
	}

	.edu-detail-hint {
		display: flex;
		align-items: center;
		margin-top: 0.4rem;
	}

	.chevron {
		width: 14px;
		height: 14px;
		color: var(--text-muted);
		transition: transform 0.35s var(--ease), color 0.3s;
	}

	.edu-row.expandable:hover .chevron {
		color: var(--accent);
	}

	.edu-row.expanded .chevron {
		transform: rotate(180deg);
	}

	.edu-expand {
		display: grid;
		grid-template-rows: 0fr;
		transition: grid-template-rows 0.4s var(--ease);
	}

	.edu-row.expanded .edu-expand {
		grid-template-rows: 1fr;
	}

	.edu-expand-inner {
		overflow: hidden;
		min-height: 0;
	}

	.edu-gpa {
		font-size: var(--text-md);
		color: var(--heading);
		font-weight: 500;
		margin-top: 0.6rem;
		transition: color 0.5s;
	}

	.edu-details {
		list-style: none;
		padding: 0.4rem 0 0;
		margin: 0;
	}

	.edu-details li {
		font-size: var(--text-base);
		color: var(--text-dim);
		line-height: 1.7;
		padding: 0.1rem 0 0.1rem 1rem;
		position: relative;
		transition: color 0.5s;
	}

	.edu-details li::before {
		content: '';
		position: absolute;
		left: 0;
		top: 0.5rem;
		width: 4px;
		height: 4px;
		border-radius: 50%;
		background: var(--border-light);
	}

	.edu-details :global(a) {
		color: var(--accent);
		text-decoration: none;
		border-bottom: 1px solid var(--accent-soft);
		transition: border-color 0.3s, color 0.5s;
	}

	.edu-details :global(a:hover) {
		border-color: var(--accent);
	}

	/* Awards */
	.award-row {
		display: grid;
		grid-template-columns: 40px 1fr;
		gap: 1.2rem;
		padding: 1.2rem 0;
		border-bottom: 1px solid var(--border);
		transition: all 0.3s;
	}

	.award-row:last-of-type {
		border-bottom: none;
	}

	.award-row:hover {
		padding-left: 0.3rem;
	}

	.award-place {
		font-family: var(--font-head);
		font-weight: 300;
		font-size: var(--text-3xl);
		color: var(--accent);
		font-style: italic;
		transition: color 0.5s;
	}

	.award-info h3 {
		font-family: var(--font-head);
		font-weight: 500;
		font-size: var(--text-xl);
		color: var(--heading);
		margin-bottom: 0.1rem;
		transition: color 0.5s;
	}

	.award-info p {
		font-size: var(--text-md);
		color: var(--text-dim);
		transition: color 0.5s;
	}

	/* Projects grid */
	.projects-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1.2rem;
	}

	/* Skills spec */
	.skills-spec {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 3rem;
	}

	.spec-group h3 {
		font-size: var(--text-base);
		font-weight: 500;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		color: var(--text-muted);
		margin-bottom: 1.2rem;
		padding-bottom: 0.5rem;
		border-bottom: 1px solid var(--border);
		transition: var(--theme-transition);
	}

	/* Gallery CTA */
	.gallery-band {
		border: 1px solid var(--border);
		padding: 2.5rem 3rem;
		display: flex;
		justify-content: space-between;
		align-items: center;
		transition: var(--theme-transition);
		position: relative;
		overflow: hidden;
	}

	.gallery-band:hover {
		border-color: var(--border-light);
	}

	.gallery-band h3 {
		font-family: var(--font-head);
		font-weight: 400;
		font-size: var(--text-3xl);
		color: var(--heading);
		margin-bottom: 0.3rem;
		transition: color 0.5s;
	}

	.gallery-band p {
		font-size: var(--text-lg);
		color: var(--text-dim);
		max-width: 360px;
		transition: color 0.5s;
	}

	.gallery-link {
		font-size: var(--text-sm);
		letter-spacing: 0.08em;
		color: var(--accent);
		text-decoration: none;
		padding: 0.5rem 1.2rem;
		border: 1px solid var(--accent-soft);
		transition: all 0.3s;
		display: flex;
		align-items: center;
		gap: 0.5rem;
		white-space: nowrap;
	}

	.gallery-link:hover {
		background: var(--accent-dim);
		border-color: var(--accent);
	}

	.gallery-link svg {
		width: 14px;
		height: 14px;
		stroke: currentColor;
		fill: none;
		stroke-width: 2;
		transition: transform 0.3s;
	}

	.gallery-link:hover svg {
		transform: translateX(3px);
	}

	@media (max-width: 768px) {
		.edu-row {
			grid-template-columns: 1fr;
			gap: 0.5rem;
		}

		.edu-year {
			text-align: left;
		}

		.edu-divider {
			display: none;
		}

		.skills-spec {
			grid-template-columns: 1fr;
		}

		.gallery-band {
			flex-direction: column;
			align-items: flex-start;
			gap: 1.5rem;
			padding: 2rem;
		}

		.projects-grid {
			grid-template-columns: 1fr;
		}
	}

	@media (max-width: 480px) {
		.timeline {
			padding-left: 1.5rem;
		}
	}
</style>
