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
			{ name: 'C / C++', proficient: true, level: 80 },
			{ name: 'Java', proficient: true, level: 75 },
			{ name: 'JS / TS', proficient: true, level: 70 },
			{ name: 'Rust', proficient: false, level: 55 },
			{ name: 'PostgreSQL', proficient: false, level: 50 }
		]
	},
	{
		title: 'ML & Frameworks',
		skills: [
			{ name: 'PyTorch', proficient: true, level: 95 },
			{ name: 'TensorFlow', proficient: true, level: 80 },
			{ name: 'Scikit-learn', proficient: false, level: 80 },
			{ name: 'OpenCV', proficient: false, level: 65 },
			{ name: 'Keras', proficient: false, level: 60 }
		]
	},
	{
		title: 'Web & UI',
		skills: [
			{ name: 'HTML5', proficient: true, level: 85 },
			{ name: 'CSS3 / SCSS', proficient: true, level: 82 },
			{ name: 'React', proficient: false, level: 75 },
			{ name: 'Vue.js', proficient: false, level: 68 },
			{ name: 'Svelte', proficient: false, level: 65 },
			{ name: 'Three.js', proficient: false, level: 60 },
			{ name: 'jQuery', proficient: false, level: 55 },
			{ name: 'Kivy', proficient: false, level: 45 },
			{ name: 'Ionic', proficient: false, level: 45 },
			{ name: 'JavaFX', proficient: false, level: 50 }
		]
	},
	{
		title: 'Tools',
		skills: [
			{ name: 'Git', proficient: true, level: 92 },
			{ name: 'NumPy', proficient: false, level: 75 },
			{ name: 'Polars', proficient: false, level: 70 },
			{ name: 'Pandas', proficient: false, level: 70 },
			{ name: 'Linux', proficient: false, level: 80 },
			{ name: 'Docker', proficient: true, level: 80 }
		]
	}
];
