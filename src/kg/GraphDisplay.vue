<template>
  <div>
    <h2>Knowledge Graph</h2>
    <svg ref="graph" :width="800" :height="600"></svg>
  </div>
</template>

<script>
import * as d3 from 'd3';

export default {
  props: ['graphData'],
  mounted() {
    this.drawGraph();
  },
  methods: {
    drawGraph() {
      const svg = d3.select(this.$refs.graph);
      const width = +svg.attr('width');
      const height = +svg.attr('height');

      const simulation = d3.forceSimulation(this.graphData.nodes)
        .force('link', d3.forceLink(this.graphData.links).id(d => d.id))
        .force('charge', d3.forceManyBody().strength(-400))
        .force('center', d3.forceCenter(width / 2, height / 2));

      const link = svg.append('g')
        .selectAll('line')
        .data(this.graphData.links)
        .enter().append('line')
        .attr('stroke-width', 2)
        .attr('stroke', '#999');

      const node = svg.append('g')
        .selectAll('circle')
        .data(this.graphData.nodes)
        .enter().append('circle')
        .attr('r', 10)
        .attr('fill', '#69b3a2')
        .call(d3.drag()
          .on('start', dragstarted)
          .on('drag', dragged)
          .on('end', dragended));

      node.append('title')
        .text(d => d.id);

      simulation
        .nodes(this.graphData.nodes)
        .on('tick', ticked);

      simulation.force('link')
        .links(this.graphData.links);

      function ticked() {
        link
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y);

        node
          .attr('cx', d => d.x)
          .attr('cy', d => d.y);
      }

      function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }

      function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
      }

      function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }
    },
  },
};
</script>
