/* Search core for "Ask this site": BM25 over a static inverted index. Same tokenizer as tools/site_ai/build_index.py. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.SiteSearch = factory();
})(this, function () {
  var STOP = new Set("a an the of and or to in on for with is are was were be been it its this that these those as at by from we you i not but if then so than which who what when where how do does did can could should would will may might has have had into about over under also".split(" "));
  var WORD = /[a-z0-9]+(?:[_.][a-z0-9]+)*/g;
  var SUF = ["ations", "ation", "ings", "ing", "ies", "ied", "ed", "es", "s"];

  function stem(w) {
    for (var i = 0; i < SUF.length; i++) {
      var s = SUF[i];
      if (w.endsWith(s) && w.length - s.length >= 3) return w.slice(0, -s.length) + (s === "ies" || s === "ied" ? "y" : "");
    }
    return w;
  }
  function tokens(text) {
    return (String(text).toLowerCase().match(WORD) || []).filter(function (w) { return !STOP.has(w); }).map(stem);
  }

  /* idx: {n, avg, len[], inv{term:[id,tf,id,tf...]}, chunks[[url,title,heading,text]...], faq_from} */
  function search(idx, query, k) {
    k = k || 5;
    var q = Array.from(new Set(tokens(query)));
    if (!q.length) return [];
    var N = idx.n, score = new Float64Array(N), hit = new Uint8Array(N);
    q.forEach(function (term) {
      var post = idx.inv[term];
      if (!post) return;
      var df = post.length / 2, idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
      for (var j = 0; j < post.length; j += 2) {
        var id = post[j], tf = post[j + 1], L = idx.len[id];
        score[id] += idf * tf * 2.4 / (tf + 1.4 * (0.25 + 0.75 * L / idx.avg));
        hit[id]++;
      }
    });
    var ids = [];
    for (var i = 0; i < N; i++) if (hit[i]) {
      score[i] *= (0.4 + 0.6 * hit[i] / q.length) * (i >= idx.faq_from ? 1.6 : 1);
      ids.push(i);
    }
    ids.sort(function (a, b) { return score[b] - score[a]; });
    return ids.slice(0, k).map(function (id) {
      var c = idx.chunks[id];
      return { id: id, score: score[id], url: c[0], title: c[1], heading: c[2], text: c[3], faq: id >= idx.faq_from, matched: hit[id], of: q.length };
    });
  }

  function sentences(text) {
    return String(text).split(/(?<=[.!?])\s+(?=[A-Z(\\])/).filter(function (s) { return s.trim().length > 0; });
  }
  /* The n sentences of `text` that share the most query terms, kept in their original order. */
  function bestSentences(text, query, n) {
    n = n || 2;
    var q = new Set(tokens(query)), ss = sentences(text);
    var scored = ss.map(function (s, i) {
      var t = new Set(tokens(s)), m = 0;
      t.forEach(function (w) { if (q.has(w)) m++; });
      return { i: i, s: s, v: m / Math.sqrt(Math.max(t.size, 4)) };
    });
    var top = scored.filter(function (x) { return x.v > 0; }).sort(function (a, b) { return b.v - a.v; }).slice(0, n);
    if (!top.length) top = scored.slice(0, 1);
    return top.sort(function (a, b) { return a.i - b.i; }).map(function (x) { return x.s; });
  }
  return { tokens: tokens, stem: stem, search: search, sentences: sentences, bestSentences: bestSentences };
});
