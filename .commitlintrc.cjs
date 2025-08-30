// Super flexible commitlint config - accepts your natural commit style
module.exports = {
  rules: {
    // Allow many different types including your preferences  
    'type-enum': [1, 'always', [
      'feat', 'fix', 'docs', 'style', 'refactor', 'perf', 'test', 'build', 
      'ci', 'chore', 'revert', 'merge', 'update', 'add', 'remove', 'cleanup',
      'enhance', 'implement', 'complete', 'deploy', 'release', 'hotfix'
    ]], // Warning only, not error
    
    // Generous length limits (warnings, not errors)
    'header-max-length': [1, 'always', 200], // Warning at 200 chars
    'subject-min-length': [1, 'always', 3],  // Warning if less than 3
    
    // Essential requirements (errors)
    'subject-empty': [2, 'never'], // Must have subject
    'type-empty': [2, 'never'],    // Must have type
    
    // Everything else is flexible (disabled)
    'scope-enum': [0],        // Any scope allowed
    'scope-case': [0],        // Any case allowed  
    'scope-empty': [0],       // Scope optional
    'subject-case': [0],      // Any case allowed
    'subject-full-stop': [0], // Period optional
    'type-case': [0],         // Any case for type
    'header-case': [0],       // Any case for header
    'header-full-stop': [0],  // No period requirements
    
    // No body/footer restrictions
    'body-leading-blank': [0],
    'body-max-line-length': [0],
    'body-case': [0],
    'footer-leading-blank': [0],
    'footer-max-line-length': [0]
  }
};
