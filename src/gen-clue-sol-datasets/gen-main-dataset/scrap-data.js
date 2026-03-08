const fs = require('fs');
const path = require('path');

// Directory containing the JS files
const jsDirs = [
    '/home/adi/workspace/rebus/careuri-definitii/js',
    '/home/adi/workspace/rebus/careuri-definitii-2/js',
    '/home/adi/workspace/rebus/careuri-definitii-3/js',
    '/home/adi/workspace/rebus/careuri-definitii-4/js',
    '/home/adi/workspace/rebus/careuri-definitii-9/js',
    '/home/adi/workspace/rebus/careuri-definitii-gazea-1/js',
    '/home/adi/workspace/rebus/careuri-definitii-gazea-2/js',
    '/home/adi/workspace/rebus/careuri-definitii-guinea-1/js'
];

// Array to store all extracted data with full paths
const allData = [];
let processedCount = 0;
let errorCount = 0;

// Process each directory
jsDirs.forEach(jsDir => {
    if (!fs.existsSync(jsDir)) {
        console.warn(`⚠ Directory not found: ${jsDir}`);
        return;
    }

    const files = fs.readdirSync(jsDir)
        .filter(f => f.endsWith('.js') && f !== 'jquery.crossword.js' && f !== 'jquery.remchar.js');

    // Process each file in the directory
    files.forEach(file => {
        const filePath = path.join(jsDir, file);
        try {
            const content = fs.readFileSync(filePath, 'utf8');

            // Find the start of puzzleData array
            const startIdx = content.indexOf('var puzzleData');
            if (startIdx === -1) return;

            // Find the array opening bracket
            const arrayStart = content.indexOf('[', startIdx);
            if (arrayStart === -1) return;

            // Extract the array content more carefully
            let bracketCount = 0;
            let inString = false;
            let escapeNext = false;
            let endIdx = -1;

            for (let i = arrayStart; i < content.length; i++) {
                const char = content[i];

                if (escapeNext) {
                    escapeNext = false;
                    continue;
                }

                if (char === '\\') {
                    escapeNext = true;
                    continue;
                }

                if (char === "'" || char === '"' || char === '`') {
                    inString = !inString;
                    continue;
                }

                if (!inString) {
                    if (char === '[') bracketCount++;
                    else if (char === ']') {
                        bracketCount--;
                        if (bracketCount === 0) {
                            endIdx = i + 1;
                            break;
                        }
                    }
                }
            }

            if (endIdx === -1) return;

            const arrayString = content.substring(arrayStart, endIdx);

            // Convert JavaScript object notation to JSON
            let jsonString = arrayString.replace(/'/g, '"');

            // Parse the JSON
            const puzzleData = JSON.parse(jsonString);

            // Extract answer and clue from each puzzle entry
            puzzleData.forEach(item => {
                if (item.answer && item.clue) {
                    allData.push({
                        filePath: filePath,
                        answer: item.answer,
                        clue: item.clue.replace(/^(?:1[0-5]|[1-9])\)\t\t/, '').trim().replace(/^--\t\t/g, '')
                    });
                }
            });

            console.log(`✓ Processed ${file}: ${puzzleData.length} entries`);
            processedCount++;
        } catch (e) {
            console.error(`✗ Error parsing ${file}: ${e.message}`);
            errorCount++;
        }
    });
});

// Save to CSV file in the first directory
const outputDir = "." //path.dirname(jsDirs[0]);
const csvPath = path.join(outputDir, 'extracted_data.csv');
const csvHeader = 'Answer,Clue,File\n';
const csvRows = allData.map(item =>
    `"${item.answer.replace(/"/g, '""')}","${item.clue.replace(/"/g, '""').trim()}","${item.filePath.replace(/"/g, '""')}"`
).join('\n');

fs.writeFileSync(csvPath, csvHeader + csvRows);

// Also save as JSON
const jsonPath = path.join(outputDir, 'extracted_data.json');
fs.writeFileSync(jsonPath, JSON.stringify(allData, null, 2));

console.log(`\n✓ Extracted ${allData.length} entries from ${processedCount} files`);
console.log(`⚠ Errors: ${errorCount}`);
console.log(`✓ Saved to: ${csvPath}`);
console.log(`✓ Saved to: ${jsonPath}`);