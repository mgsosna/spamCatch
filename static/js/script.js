const SPAM_API = "/classify"
var spamProbTemplate = `Spam probability: <span style="color: {0}">{1}</span><span style="color:white">%</span>`;
var decisionTemplate = `<span style="color: {0}">{1}</span>`;

function onSubmit() {
    var text = d3.select("#text").property("value");
    console.log(text);
    updateSpamProb(text);
}

function updatePrintText(value) {
    var div = d3.select("#printText");
    div.text(value);
}

function updateSpamProb(value) {
    var div = d3.select("#spamProb");
    var url = `${SPAM_API}/${value}`;

    d3.json(url).then(prob => {
         var color = prob2color((1-prob));
         div.html(String.format(spamProbTemplate, color, Math.round(100*prob)));
         updateDecision(prob, color);
     });
}

function updateDecision(prob, color) {
    var div = d3.select("#decision");

    if (prob < 0.25) {
        div.html(String.format(decisionTemplate, color, "Looks like ham!"));
    }
    else if (prob > 0.25 && prob < 0.5) {
        div.html(String.format(decisionTemplate, color, "Hm... something's fishy..."));
    }
    else if (prob > 0.5) {
        div.html(String.format(decisionTemplate, color, "Look out, it's spam!"));
    }
}

// Modified from https://gist.github.com/mlocati/7210513
function prob2color(prob) {
	var r, g, b = 0;
	if(prob < 0.5) {
		r = 255;
		g = Math.round(510 * prob);
	}
	else {
		g = 255;
		r = Math.round(510 - 510 * prob);
	}
	var h = r * 0x10000 + g * 0x100 + b * 0x1;
	return '#' + ('000000' + h.toString(16)).slice(-6);
}

// Taken from https://stackoverflow.com/questions/2534803/use-of-string-format-in-javascript
String.format = function() {
      var s = arguments[0];
      for (var i = 0; i < arguments.length - 1; i++) {
          var reg = new RegExp("\\{" + i + "\\}", "gm");
          s = s.replace(reg, arguments[i + 1]);
      }
      return s;
  }

d3.select("#text").on("change", onSubmit);
d3.select("#button").on("click", onSubmit);
