<script lang="ts">
let { data } = $props();
let firstData = data.summaries[0];
let restData = data.summaries.slice(1);
</script>

<style>
.heading-title {
	font-weight: 700;
}
</style>

<div class="container">
<div class="row justify-content-center">
<div class="col-8">
<a href="/posts/{firstData.slug}" class="text-decoration-none">
<div class="card mb-4 bg-transparent border-0" style="transform: scale(1.05);">
    <div class="card-body p-4 p-md-5">
      <h5 class="heading-title card-title">{firstData.title}</h5>
      <p class="card-text text-muted medium">{firstData.summary}</p>
      <p class="card-text text-muted small">Posted {firstData.date}</p>
    </div>
</div>
</a>
</div>
{#each restData as {slug, title, date, summary}}
  <div class="col-7">
  <a href="/posts/{slug}" class="text-decoration-none">
  <div class="card mb-3">
    <div class="card-body">
      <h5 class="heading-title card-title">{title}</h5>
      <p class="card-text text-muted medium">{summary}</p>
      <p class="card-text text-muted small">Posted {date}</p>
    </div>
  </div>
  </a>
  </div>
{/each}
</div>
</div>
