// Initiatilze Pusher JavaScript library
var pusher = new Pusher('6214a344fdd8b733dfd9', {
      cluster: 'eu'
    });


Vue.component('comment', {
    props: ['comment'],
    template: `
        <div class="row comment"> 
            <div class="col-md-2">
                <img 
                   src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/User_icon_2.svg/800px-User_icon_2.svg.png" 
                   class="img-responsive" 
                   width="90" 
                   height="90"
                >
            </div>
            <div style="padding-left:20px;">
              <div class="col-md-10 comment-text text-left" v-html="comment.username">             </div>
              <div class="col-md-10 comment-text text-left" v-html="comment.comment">             </div>
            </div>
        </div>
    `
})


var app = new Vue({
    el: '#app',
    delimiters: ['[[', ']]'],
    data: {
      username: '',
      comment: '',
      comments: [],
      hateful: 0,
      offensive: 0,
      neutral: 0,
      socket_id: ""
    },
    methods: {

        updateSentiments () {
            // Initialize the mood to 0
            let [hateful, neutral, offensive] = [0, 0, 0];
            
            // loop through all comments, then get the total of each mood
            for (comment of this.comments) {
               if (comment.sentiment > 0.4) {
                  hateful++;
               } else if (comment.sentiment < 0) {
                  offensive++;
               } else {
                   neutral++;
               }
            }
            
            const total_comments = this.comments.length;
            
            // Get the percentage of each mood
            this.offensive = ((offensive/total_comments) * 100).toFixed();
            this.hateful = ((hateful/total_comments) * 100).toFixed();
            this.neutral = ((neutral/total_comments) * 100).toFixed()
            
            // Return an object of the mood values
            return {hateful, neutral, offensive}
        },
        
        addComment () {
           
            fetch("/add_comment", {
                 method: "post",
                 headers: {
                 'Accept': 'application/json',
                 'Content-Type': 'application/json'
                 },
                 body: JSON.stringify({
                     id: this.comments.length,
                     username: this.username,
                     comment: this.comment,
                     socket_id: this.socket_id
                 })
             })
             .then( response => response.json() )
             .then( data => {
                 // Add the new comment to the comments state data
                 this.comments.push({
                     id: data.id,
                     username: data.username,
                     comment: data.comment,
                     sentiment: data.sentiment
                 })
                 
                 // Update the sentiment score
                 this.updateSentiments();
              })
              
            this.username = "";
            this.comment = "";
         },
    },
    created () {
        // Set the socket ID
        pusher.connection.bind('connected', () => {
            this.socket_id = pusher.connection.socket_id;
        });
        
        // Subscribe to the live-comments channel
        var channel = pusher.subscribe('live-comments');
        
        // Bind the subscribed channel (live-comments) to the new-comment event
        channel.bind('new-comment', (data) => {
           this.comments.push(data);
           
           // Update the sentiment score
           this.updateSentiments();
        });
    },
})

