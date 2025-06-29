module.exports = {
  // Specify the root directory for Jest to find test files
  roots: ['<rootDir>/Frontend/src'],  // Updated to match your project structure

  // Use Babel to transform JavaScript and JSX files
  transform: {
    '^.+\\.[t|j]sx?$': 'babel-jest', // Use babel-jest to parse .js and .jsx files
  },

  // Ignore transformation for certain node_modules except axios
  transformIgnorePatterns: [
    "<rootDir>/node_modules/(?!axios)"  // Ensure axios can be correctly transformed
  ],
};
