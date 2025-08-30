module.exports = {
	extends: ['@commitlint/config-conventional'],
	ignores: [(message) => /^Merge\b|^Revert\b|^chore\(release\)/i.test(message)],
	rules: {
		'header-max-length': [2, 'always', 120],
		'subject-case': [0],
		'type-enum': [
			2,
			'always',
			[
				'build',
				'chore',
				'ci',
				'docs',
				'feat',
				'fix',
				'perf',
				'refactor',
				'revert',
				'style',
				'test',
				'release',
				'deps',
				'infra',
			],
		],
	},
};
