<template>
  <div>
    <h1>PDF Knowledge Graph</h1>
    <input type="file" @change="handleFileUpload" multiple />
    <button @click="submitFiles">Upload</button>
    <div v-if="graphData">
      <GraphDisplay :graphData="graphData" />
    </div>
  </div>
</template>

<script>
import axios from 'axios';
import GraphDisplay from './GraphDisplay.vue';

export default {
  data() {
    return {
      selectedFiles: null,
      graphData: null,
    };
  },
  methods: {
    handleFileUpload(event) {
      this.selectedFiles = event.target.files;
    },
    async submitFiles() {
      const formData = new FormData();
      for (let i = 0; i < this.selectedFiles.length; i++) {
        formData.append('files', this.selectedFiles[i]);
      }
      const response = await axios.post('http://localhost:3000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      this.graphData = response.data;
    },
  },
};
</script>
