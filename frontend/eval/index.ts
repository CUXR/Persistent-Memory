import * as fs from "fs";
import * as path from "path";
import { MemoryStore, type MemoryRecord } from "./memoryStore";

// This evaluation script loads test cases and memory records from JSON files, runs the retrieval function for each test case, and calculates metrics based on the results. It also logs any failed cases for further analysis.
type EvalCase = {
  query: string;
  expected_person: string;
  expected_items: string[];
};

// Metrics report structure for evaluation results.
type MetricsReport = {
  accuracy_person: number;
  avg_items_returned: number;
  pass_rate: number;
};

// Simple in-memory store implementation for evaluation. It matches the query against the person field in the memory records and returns the associated items if a match is found.
function loadJsonFile<T>(fileName: string): T {
  const filePath = path.join(process.cwd(), "eval", fileName);
  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw) as T;
}

// Main evaluation function that processes test cases, retrieves results from the memory store, and calculates metrics. It also collects failed cases for reporting.
function main(): void {
  const testCases = loadJsonFile<EvalCase[]>("cases.json");
  const memories = loadJsonFile<MemoryRecord[]>("memories.json");

  const store = new MemoryStore(memories);

  let correctPersonCount = 0;
  let totalItemsReturned = 0;
  let passedCases = 0;

  const failures: Array<{
    query: string;
    expected_person: string;
    actual_person: string | null;
    expected_items: string[];
    actual_items: string[];
  }> = [];

  for (const testCase of testCases) {
    const result = store.retrieve(testCase.query);

    const personCorrect = result.resolvedPerson === testCase.expected_person;
    if (personCorrect) {
      correctPersonCount += 1;
    }

    const matchedItems = testCase.expected_items.filter((item) =>
      result.items.includes(item)
    );

    totalItemsReturned += result.items.length;

    const itemsCorrect = matchedItems.length === testCase.expected_items.length;
    const passed = personCorrect && itemsCorrect;

    if (passed) {
      passedCases += 1;
    } else {
      failures.push({
        query: testCase.query,
        expected_person: testCase.expected_person,
        actual_person: result.resolvedPerson,
        expected_items: testCase.expected_items,
        actual_items: result.items
      });
    }
  }

  // Calculate and log the metrics report based on the evaluation results.
  const report: MetricsReport = {
    accuracy_person:
      testCases.length === 0 ? 0 : correctPersonCount / testCases.length,
    avg_items_returned:
      testCases.length === 0 ? 0 : totalItemsReturned / testCases.length,
    pass_rate: testCases.length === 0 ? 0 : passedCases / testCases.length
  };

  console.log(JSON.stringify(report, null, 2));

  if (failures.length > 0) {
    console.log("\nFailed cases:");
    console.log(JSON.stringify(failures, null, 2));
    process.exitCode = 1;
  }
}

main();