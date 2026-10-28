import assert from 'node:assert/strict';
import test from 'node:test';
import { preserveTaskCompletion, taskIdentity } from './taskChecklist.js';

test('completion is preserved for the same task despite title case and whitespace changes', () => {
  const oldTask = { title: '  Confirm   venue ', detail: 'Call the hall.' };
  const newTask = { title: 'confirm venue', detail: 'Call the hall.' };
  const completed = { [taskIdentity(oldTask)]: true };

  assert.deepEqual(preserveTaskCompletion([oldTask], completed, [newTask]), {
    [taskIdentity(newTask)]: true,
  });
});

test('new or materially changed tasks are not marked complete', () => {
  const previous = [
    { title: 'Confirm venue', detail: 'Call the hall.' },
    { title: 'Send invitations', detail: 'Invite the guest list.' },
  ];
  const completed = { [taskIdentity(previous[0])]: true };
  const next = [
    { title: 'Confirm venue', detail: 'Compare contract terms.' },
    { title: 'Book transport', detail: 'Arrange accessible transport.' },
  ];

  assert.deepEqual(preserveTaskCompletion(previous, completed, next), {});
});
