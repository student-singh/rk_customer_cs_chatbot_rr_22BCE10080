const { spawn } = require('child_process');

async function invokeRAG(query, history) {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', ['../rag_query.py']);  // Relative to backend/
    const inputData = JSON.stringify({ query, history });
    pythonProcess.stdin.write(inputData);
    pythonProcess.stdin.end();

    let output = '';
    pythonProcess.stdout.on('data', (data) => { output += data.toString(); });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error('RAG process failed'));
      } else {
        try {
          const response = JSON.parse(output.trim());
          resolve(response);
        } catch (parseErr) {
          reject(new Error('Invalid RAG JSON'));
        }
      }
    });

    pythonProcess.on('error', reject);
  });
}

module.exports = { invokeRAG };