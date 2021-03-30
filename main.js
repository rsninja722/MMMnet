var truth = [0, 3, 5, 6, 8, 10, 12, 14, 16, 18, 21, 22, 24, 26, 29, 30, 32, 34, 37, 38, 41, 42, 45, 46, 48, 50, 52, 54, 56, 58, 61, 62, 3, 6, 10, 14, 16, 22, 26, 30, 32, 37, 42, 45, 48, 54, 58, 62, 6, 14, 16, 26, 32, 42, 48, 58, 6, 16, 42, 48, 16, 42, 16];
var bestModel = ["[[],-1,[],-0.16002717570714498]","[[],-1,[],-0.14268313964132]","[[],-1,[],0.8014942014227368]","[[],-1,[],0.30194905643109116]","[[],-1,[],-0.08724278621506906]","[[],-1,[],-0.1770481944608485]","[[],-1,[],0.9891180173269978]","[[],-1,[],-0.4978679570241108]","[[],-1,[],0.4768982041705929]","[[],0,[],-0.16002717570714498]","[[],1,[],-0.14268313964132]","[[],2,[],0.8014942014227368]","[[],3,[],0.30194905643109116]","[[],4,[],-0.08724278621506906]","[[],5,[],-0.1770481944608485]","[[],6,[],0.9891180173269978]","[[],7,[],-0.4978679570241108]","[[],8,[],0.4768982041705929]","[[],-1,[],-0.735429062765261]","[[0,9],-1,[-0.22886220561149462,0.24196831440598698],0.225187212057651]","[[1,10],-1,[-0.25635014639619125,0.3579675527899353],-0.847939464986762]","[[2,11],-1,[0.15759657676179084,0.4643614044475479],-0.0002895011739307077]","[[3,12],-1,[0.0727166790981586,0.09073422410959009],0.11947143726500623]","[[4,13],-1,[0.08071936625130482,0.4004493274906116],-0.49745736562017606]","[[5,14],-1,[0.400036760202523,-0.24409053188934582],0.17574621897354378]","[[6,15],-1,[-0.4936224031031464,0.1492066153980578],0.1918768076828218]","[[7,16],-1,[0.3391891298812024,0.21785231867529165],2.8452309714214725]","[[8,17],-1,[-0.22090543577958854,0.1500338266004866],-1.2910603484815535]","[[18,27],-1,[-0.11435958018984604,0.16590709934272851],-0.052185526582671046]","[[19,21,22,23],-1,[-0.33448207836866906,-0.49057538890612284,-0.1308297558150313,0.31091828059322113],-0.6318774123472959]","[[20,24,25,26,28,29],-1,[-0.3248032997658046,0.3284930491054574,-0.22400370314719642,-0.31191544113669334,0.2710549476944242,0.41150750038901074],0.02143834078823441]"];
var training;
var models = [];
var currentModel = -1;
var isFirstRound = 1;

var record = 0;

class node {
    constructor(inputs, linkWeightsTo = -1, weights = [], bias = (Math.random() * 2 - 1)) {
        this.inputs = inputs;
        this.weights = weights;
        if(weights.length === 0) {
            for (var i = 0; i < inputs.length; i++) {
                this.weights.push(Math.random() - 0.5);
            }
        }       

        this.bias = bias;
        this.linkWeightsTo = linkWeightsTo;
        if (linkWeightsTo > -1) {
            this.weights = models[currentModel][linkWeightsTo].weights;
            this.bias = models[currentModel][linkWeightsTo].bias;
        }

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
                sum += models[currentModel][this.inputs[i]].output * this.weights[i];
            }
            sum += this.bias;

            this.outputCache = this.sigmoid(sum);

            return this.outputCache;
        }
    }

    get data() {
        return JSON.stringify([this.inputs,this.linkWeightsTo,this.weights,this.bias]);
    }

    set data(str) {
        var d = JSON.parse(str);
        this.inputs = d[0];
        this.linkWeightsTo = d[1];
        this.weights = d[2];
        this.bias = d[3];
    }
}

function saveModel(index) {
    var arr = [];
    for(var i=0;i<models[index].length;i++) {
        arr.push(models[index][i].data);
    }
    return JSON.stringify(arr);
}

function loadModel(arr,index) {
    for(var i=0;i<models[index].length;i++) {
        models[index][i].data = arr[i];
    }
}

var bracketData;

function initiateNodes() {
    for (var i = 0; i < 9; i++) {
        // animal a
        models[currentModel].push(new node([]));
    }
    for (var i = 0; i < 9; i++) {
        // animal b
        models[currentModel].push(new node([], i));
    }
    models[currentModel].push(new node([])); // is first round
    for (var i = 0; i < 9; i++) {
        models[currentModel].push(new node([i, i + 9]));
    }
    models[currentModel].push(new node([18, 27]));
    models[currentModel].push(new node([19, 21, 22, 23]));

    models[currentModel].push(new node([20, 24, 25, 26, 28, 29]));
}

function compareMammals(a, b) {
    for (var i = 0; i < 9; i++) {
        models[currentModel][i].outputCache = bracketData[a][i + 1];
    }
    for (var i = 0; i < 9; i++) {
        models[currentModel][i + 9].outputCache = bracketData[b][i + 1];
    }
    models[currentModel][18].outputCache = isFirstRound;
    for (var i = 19; i < 31; i++) {
        models[currentModel][i].outputCache = undefined;
    }
    return models[currentModel][30].output;
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
        for (var j = 0; j < 9; j++) {
            bracketData[i][j + 1] = parseInt(bracketData[i][j + 1]);
        }
        if (bracketData[i][10].trim() !== "") {
            bracketData[i][10] = bracketData[i][10].split(".");
            for (var j = 0; j < bracketData[i][10].length; j++) {
                bracketData[i][10][j] = parseInt(bracketData[i][10][j]) - 2;
            }
        } else {
            bracketData[i][10] = [];
        }

        bracketData[i].push(i);
    }

    for (var i = 0; i < 100; i++) {
        currentModel++;
        models.push([]);
        initiateNodes();
    }

    // loadModel(bestModel,0);

    training = setInterval(train, 4);

    // console.log(generateBracket());
}

function train() {
    var start = window.performance.now();
    while(window.performance.now() - start < 4) {
        var best = { score: 0, index: 0 };

        var scores = [];

        for (var i = 0; i < 100; i++) {
            currentModel = i;
            var score = compareBrackets(guessBracket(), truth);
            if (score > best.score) {
                best.index = i;
                best.score = score;
            }
            scores.push([i, score]);
        }

        scores.sort((a, b) => b[1] - a[1]);

        for (var i = 25; i < 100; i++) {
            currentModel = scores[i][0];
            var shouldCopy = Math.round(Math.random()*5) === 0;
            for (var j = 0; j < models[currentModel].length; j++) {
                var n = models[currentModel][j];

                if (n.linkWeightsTo > -1) {
                    n.weights = models[currentModel][n.linkWeightsTo].weights;
                    n.bias = models[currentModel][n.linkWeightsTo].bias;
                } else {
                    if(shouldCopy) {
                        n.weights = models[scores[i%25][0]][j].weights.slice(0);
                        n.bias = models[scores[i%25][0]][j].bias;
                    }
                    n.weights.map((a) => a + ((Math.random() - 0.5) ) / (75 - i));
                    n.bias += ((Math.random() - 0.5) ) / (75 - i);
                }
            }
        }

        if (best.score > record) {
            record = best.score;
            document.getElementById("records").innerText += `model: ${best.index} score: ${best.score}/63\n`;
        }

        document.getElementById("score").innerText = `model: ${best.index} score: ${best.score}/63 time: ${window.performance.now() - start}`;
    }
}

function compareBrackets(a, b) {
    var count = 0;
    for (var i = 0; i < a.length; i++) {
        if (a[i] === b[i]) {
            count++;
        }
    }
    return count;
}

var indices = new Array(64).fill().map((a, b) => b);
function guessBracket() {
    var arr = [];
    var buffers = [indices.slice(0), []];
    var readBuffer = 0;
    var writeBuffer = 1;
    isFirstRound = 1;

    var round = 0;
    while (buffers[readBuffer].length > 1) {
        buffers[writeBuffer] = [];
        var read = buffers[readBuffer];
        var write = buffers[writeBuffer];
        for (var i = 0, l = Math.ceil(read.length / 2); i < l; i++) {
            if (compareMammals(read[i * 2], read[i * 2 + 1]) > 0.5) {
                arr.push(read[i * 2]);
                write.push(read[i * 2]);
            } else {
                arr.push(read[i * 2 + 1]);
                write.push(read[i * 2 + 1]);
            }
        }

        var temp = readBuffer;
        readBuffer = writeBuffer;
        writeBuffer = temp;
        if(++round >= 2) {
            isFirstRound = 0;
        }
    }

    return arr;
}

function generateBracket() {
    arr = [];
    for (var i = 0; i < 6; i++) {
        var toSplice = [];
        for (var x = 0; x < Math.ceil(Math.pow(2, 6 - i) / 2); x++) {
            if (bracketData[x * 2][10].length > bracketData[x * 2 + 1][10].length) {
                arr.push(bracketData[x * 2][11]);
                bracketData[x * 2][10].splice(0, 1);
                toSplice.push(x * 2 + 1);
            } else {
                arr.push(bracketData[x * 2 + 1][11]);
                bracketData[x * 2 + 1][10].splice(0, 1);
                toSplice.push(x * 2);
            }
        }
        for (var x = toSplice.length - 1; x > -1; x--) {
            bracketData.splice(toSplice[x], 1);
        }
    }
    return arr;
}

load("data/2013.csv");
