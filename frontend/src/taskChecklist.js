function normalizeTaskText(value) {
  return String(value ?? '')
    .normalize('NFKC')
    .trim()
    .replace(/\s+/g, ' ')
    .toLocaleLowerCase('en');
}

export function taskIdentity(task) {
  return JSON.stringify([normalizeTaskText(task.title), normalizeTaskText(task.detail)]);
}

export function preserveTaskCompletion(previousTasks, completed, nextTasks) {
  const previouslyCompleted = new Set(
    previousTasks
      .filter((task) => completed[taskIdentity(task)])
      .map(taskIdentity),
  );
  return Object.fromEntries(
    nextTasks
      .filter((task) => previouslyCompleted.has(taskIdentity(task)))
      .map((task) => [taskIdentity(task), true]),
  );
}
