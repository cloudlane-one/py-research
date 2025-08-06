module.exports = {
  plugins: [
    [
      "semantic-release-gitmoji",
      {
        releaseRules: {
          major: [":boom:"],
          minor: [":sparkles:"],
          patch: [
            ":bug:",
            ":ambulance:",
            ":lock:",
            ":arrow_up:",
            ":adhesive_bandage:",
          ],
        },
      },
    ],
    "@semantic-release/github",
  ],
};
