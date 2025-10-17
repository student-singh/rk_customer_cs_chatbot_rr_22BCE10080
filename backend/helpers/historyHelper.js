function buildHistory(messages) {
  return messages.map(msg => `${msg.role.charAt(0).toUpperCase() + msg.role.slice(1)}: ${msg.content}`).join('\n');
}

module.exports = { buildHistory };