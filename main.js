class node {
    static nodes = [];

    constructor(inputs) {
        this.inputs = inputs;
        this.weights = [];
        for (var i = 0; i < inputs.length; i++) {
            this.weights.push(Math.random() * 2 - 1);
        }

        this.bias = Math.random() * 2 - 1;

        this.outputCache = undefined;
    }

    sigmoid(z) {
        return 1 / (1 + Math.exp(-z));
    }

    get output() {
        if (this.outputCache !== undefined) {
            return this.outputCache;
        }

        if (this.inputs.length > 0) {
            var sum = 0;
            for (var i = 0; i < this.inputs.length; i++) {
                sum += node.nodes[this.inputs[i]].output;
            }
            sum += this.bias;

            this.outputCache = this.sigmoid(sum);

            return this.outputCache;
        }
    }
}

var bracketData;

function initiateNodes() {
    for (var i = 0; i < 16; i++) {
        node.nodes.push(new node([]));
    }
    for (var i = 0; i < 16; i++) {
        node.nodes.push(new node(new Array(16).fill().map((e, i) => i)));
    }
    node.nodes.push(new node(new Array(16).fill().map((e, i) => i + 16)));
}

function compareMammals(a, b) {
    for (var i = 0; i < 8; i++) {
        node.nodes[i].outputCache = bracketData[a][i + 1];
    }
    for (var i = 0; i < 8; i++) {
        node.nodes[i + 8].outputCache = bracketData[b][i + 1];
    }
    console.log(node.nodes[node.nodes.length - 1].output);
}

function load(file) {
    fetch(file)
        .then((response) => response.text())
        .then((data) => parseBracket(data));
}

function parseBracket(data) {
    data = data.split("\n");
    data.shift();
    bracketData = [];
    for (var i = 0; i < data.length; i++) {
        bracketData.push(data[i].split(","));
    }
    for (var i = 0; i < bracketData.length; i++) {
        for (var j = 0; j < 8; j++) {
            bracketData[i][j + 1] = parseInt(bracketData[i][j + 1]);
        }
        if (bracketData[i][9] !== "" && bracketData[i][9] !== " ") {
            bracketData[i][9] = bracketData[i][9].split(".");
            for (var j = 0; j < bracketData[i][9].length; j++) {
                bracketData[i][9][j] = parseInt(bracketData[i][9][j]) - 2;
                if (isNaN(bracketData[i][9][j])) {
                    bracketData[i][9].splice(j, 1);
                }
            }
        }
    }

    initiateNodes();
    compareMammals(0,1);
}

load("data/2013.csv");
