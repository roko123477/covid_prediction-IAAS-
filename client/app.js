function getdayValue() {
    var uiday = document.getElementsByName("uiday");
    for(var i in uiday) {
      if(uiday[i].checked) {
          return parseInt(i);
      }
    }
    return -1; // Invalid Value
  }
  
  function getyearValue() {
    var uiyear = document.getElementsByName("uiyear");
    for(var i in uiyear) {
      if(uiyear[i].checked) {
          return parseInt(i);
      }
    }
    return -1; // Invalid Value
  }
  
  function onClickedcovid19() {
    console.log("Estimate price button clicked");
    var confirmed = document.getElementById("uiconfirm");
    var year = getyearValue();
    var day = getdayValue();
    var state = document.getElementById("uistate");
    var estcovid = document.getElementById("uipredictcovid19");
  
    //var url = "http://127.0.0.1:5000/predict_covid19"; //Use this if you are NOT using nginx which is first 7 tutorials
      var url = "/api/predict_covid19"
  
    $.post(url, {
        confirmed: parseInt(confirmed.value),
        year: year,
        day: day,
        state: state.value
    },function(data, status) {
        console.log(data.covid_19_cured_cases);
        estcovid.innerHTML = "<h3>" +"the approx cured patients will be :"+ data.covid_19_cured_cases.toString() ;
        console.log(status);
    });
  }
  
  function onPageLoad() {
    console.log( "document loaded" );
    //var url = "http://127.0.0.1:5000/get_state_names"; // Use this if you are NOT using nginx which is first 7 tutorials
    var url ="/api/get_state_names"// Use this if  you are using nginx. i.e tutorial 8 and onwards
    $.get(url,function(data, status) {
        console.log("got response for get_state_names request");
        if(data) {
            var state = data.state;
            var uistate = document.getElementById("uistate");
            $('#uistate').empty();
            for(var i in state) {
                var opt = new Option(state[i]);
                $('#uistate').append(opt);
            }
        }
    });
  }
  
  window.onload = onPageLoad;