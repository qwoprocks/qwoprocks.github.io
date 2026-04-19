export interface Experience {
	id?: string;
	title: string;
	company: string;
	period: string;
	location: string;
	tag: string;
	desc: string;
	bullets: string[];
}

export const experiences: Experience[] = [
	{
		title: 'Quantitative Developer',
		company: 'DRW',
		period: 'Oct 2025 – Present',
		location: 'Singapore',
		tag: 'Full-time',
		desc: 'Quantitative development spanning compute-graph frameworks, trading systems, and internal tooling.',
		bullets: [
			'Extended a Python compute-graph framework with C++ bindings (nanobind), adding cycle detection, benchmarking, GDB debugging, and improved Python typing; achieved 2–20x speedup on graph operations',
			'Built a fault-tolerant intraday fill ingestion pipeline over TCP websocket with SOD-replay-based restart recovery, ensuring consistent live position state across disconnects',
			'Extended a real-time pricing and execution-parameter engine to support China markets, integrating market data, trader signals, position/locate feeds, and static reference data; also supported feature requests from crypto and long-short rebalancing desks',
			'Built a Hyperliquid liquidation monitor (Dash + Redis) tracking ~20k wallets with distance-to-liquidation metrics + liquidation heatmaps',
			'Built an internal Slack support bot using the Claude agent SDK with semantic search over 8 codebases, auto-generated Bloomberg field knowledge, and MCP-gated access to live logs and data streams',
			'Reduced CI build times by ~20% via build-environment caching and compression tuning'
		]
	},
	{
		title: 'Cyber Security Engineer',
		company: 'DSO National Laboratories',
		period: 'Aug 2024 – Sep 2025',
		location: 'Singapore',
		tag: 'Full-time',
		desc: 'Low-level systems work on QEMU/KVM and reverse engineering tooling.',
		bullets: [
			'Low-level modifications of QEMU/KVM source code to support Intel Processor Trace and other functionalities',
			'Developed a Ghidra plugin to perform Struct and Enum recovery from disassembled code'
		]
	},
	{
		title: 'Quantitative Developer Intern',
		company: 'DRW',
		period: 'May 2024 – Aug 2024',
		location: 'Singapore',
		tag: 'Internship',
		desc: 'Parallelized Python codebases and built regression testing frameworks for quant research.',
		bullets: [
			'Parallelized and refactored several parts of a Python code base used to support Quantitative Researchers, increasing scalability and reducing latency by 30%',
			'Found and fixed a high-latency-impact, low-reproducibility bug that occurred in production',
			'Developed a framework for regression testing and reproducing production bugs in a semi-realistic setting',
			"Supported other team's C++ operations, and wrote small scripts to automate processes"
		]
	},
	{
		id: 'teaching-assistant-2',
		title: 'Teaching Assistant',
		company: 'Carnegie Mellon University',
		period: 'Jan 2024 – May 2024',
		location: 'Pittsburgh, PA',
		tag: 'Teaching',
		desc: 'TA for 10-725 Convex Optimization, supporting a cohort of 80 students.',
		bullets: [
			'Hosted weekly Office Hours for a cohort of 80 students',
			'Created exam and homework questions for the course',
			'Graded homework assignments and gave personalized feedback on mistakes made'
		]
	},
	{
		title: 'Software Engineer Intern (C++)',
		company: 'Defence Science and Technology Agency',
		period: 'Jan 2023 – Jun 2023',
		location: 'Singapore',
		tag: 'Internship',
		desc: 'Computer vision algorithms for drone detection and autonomous drone-vs-drone systems.',
		bullets: [
			'Improve in-house computer vision algorithms for robust detection and locating of drones in various environments',
			'Modify open-source code to fit client use cases, integrating it with various drone hardware and the existing code base',
			'Work on improving UI/UX for current user-facing software',
			'Develop drone algorithms for drone-vs-drone purposes (taking down another drone using a drone)',
			'Conduct on-the-ground experiments to determine the efficacy of different approaches'
		]
	},
	{
		title: 'Machine Learning Intern',
		company: 'Samsung',
		period: 'Jun 2022 – Aug 2022',
		location: 'Singapore',
		tag: 'Internship',
		desc: 'Deepfake detection visualization software with gradient-weighted class activation mapping.',
		bullets: [
			'Developed a containerized desktop and web based visualization software',
			"Integrated various image and voice Deepfake detection models into the software, visualizing their gradient-weighted class activation mapping, enhancing the explainability of the team's artificial intelligence products",
			"Provided technical input on which direction the team should head in and how to improve their current solutions, which helped shape the team's decisions"
		]
	},
	{
		title: 'Machine Learning Intern',
		company: 'Defence Science and Technology Agency',
		period: 'Jun 2021 – Aug 2021',
		location: 'Singapore',
		tag: 'Internship',
		desc: 'End-to-end ASR pipeline development with distributed training on ClearML.',
		bullets: [
			'Thoroughly tested several state-of-the-art Automatic Speech Recognition (ASR) models to verify their suitability for deployment',
			'Implemented a containerized end-to-end ASR pipeline with detailed documentation, allowing users to easily utilize the pipeline for their needs',
			'Deployed the completed pipeline to distributed systems using ClearML, enabling fast distributed training on large internal datasets'
		]
	},
	{
		id: 'teaching-assistant-1',
		title: 'Teaching Assistant',
		company: 'National University of Singapore',
		period: 'Jan 2020 – May 2023',
		location: 'Singapore',
		tag: 'Teaching',
		desc: 'TA for 6 modules over 6 semesters; nominated for teaching awards by 9 students.',
		bullets: [
			'CS2109S Introduction to AI and Machine Learning',
			'CS3241 Computer Graphics',
			'CS2106 Introduction to Operating Systems',
			'CS2040S Data Structures and Algorithms',
			'CS2103T Software Engineering',
			'CS2030 Programming Methodology II',
			'Nominated for teaching awards by 9 students'
		]
	}
];
