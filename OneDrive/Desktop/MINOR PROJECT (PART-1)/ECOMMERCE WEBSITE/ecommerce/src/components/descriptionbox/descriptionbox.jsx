import React from "react";
import "./descriptionbox.css";

export default function descriptionbox(props) {
  return (
    <div className="descriptionbox">
      <div className="descriptionbox-navigator">
        <div className="descriptionbox-nav-box">Description</div>
        <div className="descriptionbox-nav-box fade">Reviews(122)</div>
      </div>
      <div className="descriptionbox-description">
        <p>
          An ecommerce website is an online platform that is used for buying and
          seling products or services over the internet.with the help of
          ecommerce website , it is easy to buy anything from home.
        </p>
        <p>
          ecommerce website typically display products or servicesand their
          detailed description. Lorem ipsum dolor, sit amet consectetur
          adipisicing elit. Quia ducimus rem dignissimos libero sint illum totam
          voluptates, exercitationem fuga deserunt distinctio, impedit animi
          rerum laudantium minus fugiat magnam adipisci qui.
        </p>
      </div>
    </div>
  );
}
