class Home{
    constructor(){
        this.playersContainer = document.getElementById('players');
        this.getPlayers();
    }

    getPlayers(){
        const xhr = new XMLHttpRequest();
        xhr.withCredentials = true;

        const self = this;
        xhr.addEventListener('readystatechange', function () {
            if (this.readyState === this.DONE) {
                self.players = JSON.parse(this.responseText);
                self.renderPlayers();
            }
        });

        xhr.open('GET', API_URL+'players');
        xhr.setRequestHeader('authorization', 'Basic '+API_KEY);
        xhr.setRequestHeader('content-type', 'application/json');

        xhr.send();
    }

    renderPlayers(){
        this.playersContainer.innerHTML = '';
        const sorted = [...this.players["players"]].sort((a, b) => (a.stars ?? 0) - (b.stars ?? 0));
        sorted.forEach(player => {
            const el = document.createElement('div');
            el.className = 'player';

            const name = document.createElement('div');
            name.className = 'name';
            name.textContent = player.name;

            const description = document.createElement('div');
            description.className = 'description';
            description.textContent = player.description ?? '';

            const stars = document.createElement('div');
            stars.className = 'stars';
            const total = 5;
            const filled = player.stars ?? 0;
            for (let i = 1; i <= total; i++) {
                const star = document.createElement('span');
                star.className = 'star' + (i <= filled ? ' filled' : '');
                star.textContent = '★';
                stars.appendChild(star);
            }

            el.appendChild(name);
            el.appendChild(description);
            el.appendChild(stars);

            el.addEventListener('click', () => {
                window.location.href = `/game?player=${encodeURIComponent(player.uid)}`;
            });

            this.playersContainer.appendChild(el);
        });
    }
}

document.addEventListener('DOMContentLoaded', function() {
    new Home();
});
