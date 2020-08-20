var nzh = require("nzh/cn");
var fs = require("fs");

/************** Settings ******************/
var n = 100000;
var mode = 'money'; // money or number
var maxNumDigit = 7;
var fileName = 'money.txt';
/*****************************************/

function random(max) {
    return Math.floor(Math.random() * Math.floor(max));
}

function append(fileName, number) {
    // the variable to determine the value of 
    // tenMin in number mode or complete in money mode
    var flag = random(1); 
    var content = '';
    if (mode === 'money') {
        content = nzh.toMoney(number / 10.0, {complete: flag})
    } else if (mode === 'number') {
        content = nzh.encodeS(number, {tenMin: flag});
    } else {
        throw new Error('Invalid Mode');
    }
    fs.appendFileSync('./' + fileName, content + '\n');
}

function start() {
    for (var i = 0; i < n; i ++) {
        var numDigit = random(maxNumDigit) + 1;
        var number = 0;
        for (var j = numDigit; j > 0; j--) {
            var digit = random(9);
            // if the first digit is 0, regenerate
            if (j == numDigit && digit == 0) {
                do {
                    digit = random(9);
                } while (digit == 0)         
            }
            // if the generated digit is 1, 2, 3,
            // the final number will be (1 * 100 + 2 * 10 + 3 *1)
            number = number + digit * Math.pow(10, j - 1);
        }
        append(fileName, number);
    }
    console.log('writing complete');
}

// start();


function transNumber(number, type_){
    var content = '';
    if (type_ == 'money'){
        content = nzh.toMoney(number);
    } else if (type_ == 'number') {
        content = nzh.encodeS(number);
    } else {
        throw new Error('Invalid Mode');
    }
    return content;
}