export interface Skill {
	name: string;
	proficient: boolean;
	level: number; // percentage width for progress bar
}

export interface SkillGroup {
	title: string;
	skills: Skill[];
}

export const skillGroups: SkillGroup[] = [
	{
		title: 'Languages',
		skills: [
			{ name: 'Python', proficient: true, level: 95 },
			{ name: 'Java', proficient: true, level: 88 },
			{ name: 'C / C++', proficient: true, level: 85 },
			{ name: 'JS / TS', proficient: true, level: 82 },
			{ name: 'C#', proficient: false, level: 50 },
			{ name: 'Rust', proficient: false, level: 55 },
			{ name: 'PHP', proficient: false, level: 40 },
			{ name: 'PostgreSQL', proficient: false, level: 50 }
		]
	},
	{
		title: 'ML & Frameworks',
		skills: [
			{ name: 'PyTorch', proficient: true, level: 95 },
			{ name: 'TensorFlow', proficient: true, level: 85 },
			{ name: 'Scikit-learn', proficient: false, level: 70 },
			{ name: 'OpenCV', proficient: false, level: 65 },
			{ name: 'Keras', proficient: false, level: 60 }
		]
	},
	{
		title: 'Web',
		skills: [
			{ name: 'React', proficient: false, level: 75 },
			{ name: 'Vue.js', proficient: false, level: 68 },
			{ name: 'Svelte', proficient: false, level: 65 },
			{ name: 'Node.js', proficient: false, level: 75 },
			{ name: 'Three.js', proficient: false, level: 60 }
		]
	},
	{
		title: 'Tools',
		skills: [
			{ name: 'Docker', proficient: true, level: 88 },
			{ name: 'Git', proficient: true, level: 92 },
			{ name: 'NumPy', proficient: false, level: 70 },
			{ name: 'Pandas', proficient: false, level: 65 },
			{ name: 'AWS', proficient: false, level: 60 },
			{ name: 'Linux', proficient: false, level: 80 }
		]
	}
];
