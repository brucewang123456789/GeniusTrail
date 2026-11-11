'use strict';
const {VoderChatViewProvider} = require('../src/chat-view.cjs');

class TestChatViewProvider extends VoderChatViewProvider {
  constructor(app, options = {}) {
    // These editor tests exercise routing; project memory has its own suite.
    const projectMemory = {
      begin: async () => '',
      finish: async () => {},
      fail: async () => {},
      end: async () => {},
      clear: () => {},
    };
    super(app, {projectMemory, ...options});
  }
}
module.exports = {TestChatViewProvider};
