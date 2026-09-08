<template>
  <div>
    <h1>VMail Inbox</h1>
    <table class="mail-table">
      <tbody>
        <tr
          v-for="email in unarchivedEmails"
          :key="email.id"
          :class="['clickable', email.read ? 'read' : '']"
          @click="email.read = !email.read"
        >
          <td>{{ email.from }}</td>
          <td>{{ email.subject }}</td>
          <td>{{ email.sentAt }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup>
const emails = ref([
  {
    id: 1,
    from: 'team@vuemastery.com',
    subject: "What's up with Vue 3.0? Here's how to find out from Evan You",
    sentAt: '2020-03-27T18:25:43.511Z',
    archived: false,
    read: true
  },
  {
    id: 2,
    from: 'jeffrey@vuetraining.net',
    subject: 'Learn by doing - Vue 3 Zero to Intermediate in 8 weeks',
    sentAt: '2020-05-20T18:25:43.511Z',
    archived: false,
    read: false
  },
  {
    id: 3,
    from: 'damian@dulisz.com',
    subject: '#177: Updated Vue.js Roadmap; Nuxt v2.12 released',
    sentAt: '2020-03-18T18:25:43.511Z',
    archived: false,
    read: false
  }
])

const sortedEmails = computed(() =>
  [...emails.value].sort((a, b) => new Date(b.sentAt) - new Date(a.sentAt))
)

const unarchivedEmails = computed(() => sortedEmails.value.filter((email) => !email.archived))
</script>

<style scoped>
.clickable {
  cursor: pointer;
}

.mail-table {
  max-width: 1000px;
  margin: auto;
  border-collapse: collapse;
}

.mail-table tr.read {
  background-color: #eee;
}

.mail-table tr {
  height: 40px;
}

.mail-table td {
  border-bottom: 1px solid black;
  padding: 5px;
  text-align: left;
}
</style>
